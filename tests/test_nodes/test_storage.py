"""
Tests for the storage layer — aiosqlite DB init, ExtractionRepository
(SCD Type 2 upsert, idempotency guard, DLQ routing), and JSON writer.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator

import aiosqlite
import pytest

from src.storage.db import get_connection, init_db
from src.storage.repository import ExtractionRepository
from src.storage.json_writer import write_json_output


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
async def tmp_db(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[str, None]:
    """Create a temporary SQLite database for testing.

    Uses ``monkeypatch`` (pytest built-in) instead of direct attribute
    mutation — this keeps state changes **scoped to the current test**
    and eliminates cross-test / cross-fixture contamination.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    from src.config import settings
    monkeypatch.setattr(settings, "sqlite_db_path", db_path)

    await init_db()
    yield db_path

    # Teardown — monkeypatch auto-restores after yield,
    # just need to clean up the file on disk.
    Path(db_path).unlink(missing_ok=True)
    Path(db_path + "-shm").unlink(missing_ok=True)
    Path(db_path + "-wal").unlink(missing_ok=True)


@pytest.fixture
def tmp_output_dir() -> str:
    """Create a temporary directory for JSON output.

    Uses ``yield`` (not ``return``) so the ``with`` block stays alive
    for the **entire test** — otherwise ``TemporaryDirectory.__exit__``
    deletes the directory before any test code touches it.
    """
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def passed_extraction_state() -> dict[str, Any]:
    """A state dict representing a successfully validated extraction."""
    return {
        "file_path": "/tmp/test_ma.txt",
        "raw_content": "Microsoft to acquire Activision for $68.7B.",
        "doc_type": "M&A",
        "validation_passed": True,
        "validation_report": "All checks passed.",
        "retry_count": 0,
        "correction_logs": [],
        "error": None,
        "max_retries": 3,
        "extracted_data": {
            "doc_id": "abc123def456",
            "metadata": {
                "source_path": "/tmp/test_ma.txt",
                "processed_at": "2026-06-01T12:00:00",
                "model_name": "deepseek-chat",
                "doc_type": "M&A",
                "pipeline_version": "1.0.0",
                "retry_count": 0,
                "extra": {},
            },
            "raw_text_snippet": "Microsoft to acquire Activision for $68.7B.",
            "acquirer": "Microsoft Corporation",
            "target": "Activision Blizzard, Inc.",
            "total_value_usd": 68700000000.0,
            "stake_percentage": 100.0,
            "payment_method": "Cash",
        },
    }


@pytest.fixture
def failed_extraction_state() -> dict[str, Any]:
    """A state dict representing a failed extraction routed to DLQ."""
    return {
        "file_path": "/tmp/test_fail.txt",
        "raw_content": "Some garbled text that can't be extracted.",
        "doc_type": "M&A",
        "validation_passed": False,
        "validation_report": "Validation FAILED...",
        "retry_count": 3,
        "correction_logs": [
            {"cycle": 1, "error_summary": "Acquirer missing", "raw_feedback": "..."},
            {"cycle": 2, "error_summary": "Target missing", "raw_feedback": "..."},
            {"cycle": 3, "error_summary": "Still missing", "raw_feedback": "..."},
        ],
        "error": "validator_node: max retries (3) exhausted.",
        "max_retries": 3,
        "extracted_data": None,
    }


# ═══════════════════════════════════════════════════════════════════
#  DB initialization
# ═══════════════════════════════════════════════════════════════════

class TestDBInit:
    async def test_init_db_creates_tables(self, tmp_db: str) -> None:
        """After init_db, the 4 core tables should exist."""
        conn = await get_connection()
        try:
            tables = await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            table_names = [r["name"] for r in tables]
            assert "extractions" in table_names
            assert "extraction_sources" in table_names
            assert "correction_logs" in table_names
            assert "dead_letter_queue" in table_names
        finally:
            await conn.close()

    async def test_init_db_is_idempotent(self, tmp_db: str) -> None:
        """Calling init_db multiple times should not error."""
        await init_db()  # second call
        await init_db()  # third call
        conn = await get_connection()
        try:
            tables = await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            assert len(tables) >= 4
        finally:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
#  Repository — Branch A (upsert extraction)
# ═══════════════════════════════════════════════════════════════════

class TestRepositoryUpsert:
    async def test_save_new_extraction(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """A new extraction should be inserted with version=1."""
        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM extractions")
            assert len(rows) == 1
            assert rows[0]["version"] == 1
            assert rows[0]["is_current"] == 1
            assert rows[0]["doc_id"] == "abc123def456"
            assert rows[0]["doc_type"] == "M&A"
        finally:
            await conn.close()

    async def test_idempotent_skip_on_same_content(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """Running the same content twice should NOT create a duplicate."""
        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)
        await repo.save_extraction(passed_extraction_state)  # same content

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM extractions")
            assert len(rows) == 1  # Still 1 — idempotent
        finally:
            await conn.close()

    async def test_scd_type2_version_increment(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """Changing content should archive old version and create v2."""
        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)

        # Modify content (simulate file edit)
        changed_state = dict(passed_extraction_state)
        changed_state["raw_content"] = "Modified: Microsoft to acquire Activision for $70B."

        # Need a new doc_id since the test's extracted_data has a fixed one
        changed_extracted = dict(changed_state["extracted_data"])
        changed_extracted["doc_id"] = "new-doc-id-789"
        changed_state["extracted_data"] = changed_extracted

        await repo.save_extraction(changed_state)

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall(
                "SELECT * FROM extractions ORDER BY version"
            )
            assert len(rows) == 2
            # v1 is archived
            assert rows[0]["version"] == 1
            assert rows[0]["is_current"] == 0
            # v2 is current
            assert rows[1]["version"] == 2
            assert rows[1]["is_current"] == 1
        finally:
            await conn.close()

    async def test_correction_logs_saved(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """Correction logs should be written to the correction_logs table."""
        state = dict(passed_extraction_state)
        state["correction_logs"] = [
            {"cycle": 1, "error_summary": "Missing acquirer", "raw_feedback": "..."},
            {"cycle": 2, "error_summary": "Still missing", "raw_feedback": "..."},
        ]

        repo = ExtractionRepository()
        await repo.save_extraction(state)

        conn = await get_connection()
        try:
            logs = await conn.execute_fetchall(
                "SELECT * FROM correction_logs ORDER BY cycle"
            )
            assert len(logs) == 2
            assert logs[0]["cycle"] == 1
            assert logs[0]["error_summary"] == "Missing acquirer"
            assert logs[1]["cycle"] == 2
        finally:
            await conn.close()

    async def test_raw_text_snippet_saved(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """Raw text snippet should be written to extraction_sources."""
        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)

        conn = await get_connection()
        try:
            sources = await conn.execute_fetchall("SELECT * FROM extraction_sources")
            assert len(sources) == 1
            assert sources[0]["doc_id"] == "abc123def456"
            assert "Microsoft" in sources[0]["raw_text_snippet"]
        finally:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
#  Repository — Branch B (Dead Letter Queue)
# ═══════════════════════════════════════════════════════════════════

class TestRepositoryDLQ:
    async def test_failed_state_goes_to_dlq(
        self, tmp_db: str, failed_extraction_state: dict[str, Any]
    ) -> None:
        """A failed state should be written to dead_letter_queue."""
        repo = ExtractionRepository()
        await repo.save_extraction(failed_extraction_state)

        conn = await get_connection()
        try:
            dlq_rows = await conn.execute_fetchall("SELECT * FROM dead_letter_queue")
            assert len(dlq_rows) == 1
            assert dlq_rows[0]["retry_count"] == 3
            assert "max retries" in dlq_rows[0]["last_error"]
        finally:
            await conn.close()

    async def test_failed_state_not_in_extractions(
        self, tmp_db: str, failed_extraction_state: dict[str, Any]
    ) -> None:
        """Failed states should NOT appear in the extractions table."""
        repo = ExtractionRepository()
        await repo.save_extraction(failed_extraction_state)

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM extractions")
            assert len(rows) == 0  # Clean — only in DLQ
            dlq_rows = await conn.execute_fetchall("SELECT * FROM dead_letter_queue")
            assert len(dlq_rows) == 1
        finally:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
#  JSON writer
# ═══════════════════════════════════════════════════════════════════

class TestJSONWriter:
    async def test_write_json_output(
        self, tmp_output_dir: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """JSON writer should create a file on disk."""
        await write_json_output(passed_extraction_state, tmp_output_dir)

        out_path = Path(tmp_output_dir) / "test_ma_result.json"
        assert out_path.is_file()

        with open(out_path) as fh:
            data = json.load(fh)
        assert data["doc_id"] == "abc123def456"
        assert data["doc_type"] == "M&A"
        assert data["validation_passed"] is True
        assert data["extracted_data"] is not None

    async def test_json_includes_correction_logs(
        self, tmp_output_dir: str, failed_extraction_state: dict[str, Any]
    ) -> None:
        """JSON output should include correction_logs for audit."""
        await write_json_output(failed_extraction_state, tmp_output_dir)

        out_path = Path(tmp_output_dir) / "test_fail_result.json"
        assert out_path.is_file()

        with open(out_path) as fh:
            data = json.load(fh)
        assert len(data["correction_logs"]) == 3
        assert data["error"] is not None

    async def test_json_output_unknown_file(
        self, tmp_output_dir: str,
    ) -> None:
        """If file_path is missing, the writer should use 'unknown' as stem."""
        state = {"file_path": None, "extracted_data": {}, "doc_type": None}
        await write_json_output(state, tmp_output_dir)

        out_path = Path(tmp_output_dir) / "unknown_result.json"
        assert out_path.is_file()


# ═══════════════════════════════════════════════════════════════════
#  Repository + JSON dual-write integration
# ═══════════════════════════════════════════════════════════════════

class TestRepositoryDualWrite:
    async def test_save_with_output_dir_writes_both(
        self,
        tmp_db: str,
        tmp_output_dir: str,
        passed_extraction_state: dict[str, Any],
    ) -> None:
        """save_extraction with output_dir should write JSON + SQLite."""
        repo = ExtractionRepository(output_dir=tmp_output_dir)
        await repo.save_extraction(passed_extraction_state)

        # SQLite
        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM extractions")
            assert len(rows) == 1
        finally:
            await conn.close()

        # JSON
        out_path = Path(tmp_output_dir) / "test_ma_result.json"
        assert out_path.is_file()

    async def test_empty_output_dir_skips_json(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """Empty output_dir should skip JSON write."""
        repo = ExtractionRepository(output_dir="")
        await repo.save_extraction(passed_extraction_state)
        # No error — logically correct
