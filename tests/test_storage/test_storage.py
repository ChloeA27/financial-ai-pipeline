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
        "error": "extraction_validator_node: max retries (3) exhausted.",
        "max_retries": 3,
        "extracted_data": None,
    }


# ═══════════════════════════════════════════════════════════════════
#  DB initialization
# ═══════════════════════════════════════════════════════════════════


class TestDBInit:
    async def test_init_db_creates_tables(self, tmp_db: str) -> None:
        """After init_db, the bronze / silver / gold tables should exist."""
        conn = await get_connection()
        try:
            tables = await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            table_names = [r["name"] for r in tables]
            assert "extractions" in table_names
            assert "documents" in table_names
            assert "step_runs" in table_names
            assert "correction_logs" in table_names
            assert "dead_letter_queue" in table_names
            assert "extraction_sources" not in table_names  # replaced by raw store
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
        changed_state["raw_content"] = (
            "Modified: Microsoft to acquire Activision for $70B."
        )

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

    async def test_scd_type2_removes_superseded_vector(
        self, tmp_db: str, passed_extraction_state: dict[str, Any], rag_on: None
    ) -> None:
        """An archived version must stop being served as RAG context."""
        from src.rag.chroma_client import get_or_create_collection

        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)
        assert get_or_create_collection("M&A").get(ids=["abc123def456"])["ids"] == [
            "abc123def456"
        ]

        changed = dict(passed_extraction_state)
        changed["raw_content"] = "Modified: Microsoft to acquire Activision for $70B."
        changed["extracted_data"] = {
            **changed["extracted_data"],
            "doc_id": "new-doc-id-789",
        }
        await repo.save_extraction(changed)

        col = get_or_create_collection("M&A")
        assert col.get(ids=["abc123def456"])["ids"] == []  # v1 gone from index
        assert col.get(ids=["new-doc-id-789"])["ids"] == ["new-doc-id-789"]
        assert col.count() == 1

        # ...but SQLite keeps v1 as history
        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall(
                "SELECT doc_id, is_current FROM extractions ORDER BY version"
            )
            assert [(r["doc_id"], r["is_current"]) for r in rows] == [
                ("abc123def456", 0),
                ("new-doc-id-789", 1),
            ]
        finally:
            await conn.close()

    async def test_idempotent_rerun_keeps_vector(
        self, tmp_db: str, passed_extraction_state: dict[str, Any], rag_on: None
    ) -> None:
        from src.rag.chroma_client import get_or_create_collection

        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)
        await repo.save_extraction(passed_extraction_state)
        assert get_or_create_collection("M&A").count() == 1

    async def test_other_files_vectors_untouched(
        self, tmp_db: str, passed_extraction_state: dict[str, Any], rag_on: None
    ) -> None:
        from src.rag.chroma_client import get_or_create_collection

        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)

        other = dict(passed_extraction_state)
        other["file_path"] = "/tmp/another_deal.txt"
        other["raw_content"] = "Totally different deal text."
        other["extracted_data"] = {
            **other["extracted_data"],
            "doc_id": "other-doc",
            "metadata": {
                **other["extracted_data"]["metadata"],
                "source_path": "/tmp/another_deal.txt",
            },
        }
        await repo.save_extraction(other)

        assert sorted(get_or_create_collection("M&A").get()["ids"]) == [
            "abc123def456",
            "other-doc",
        ]

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

    async def test_full_source_text_is_kept_in_the_raw_store(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        """The whole text (not a 500-char snippet) is stored and registered."""
        from src.storage.raw_store import read_raw

        repo = ExtractionRepository()
        await repo.save_extraction(passed_extraction_state)

        conn = await get_connection()
        try:
            docs = await conn.execute_fetchall("SELECT * FROM documents")
        finally:
            await conn.close()
        assert len(docs) == 1
        assert docs[0]["last_status"] == "stored"
        text = await read_raw(docs[0]["raw_uri"])
        assert text == passed_extraction_state["raw_content"]


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

    async def test_repeated_failure_of_same_content_is_one_row(
        self, tmp_db: str, failed_extraction_state: dict[str, Any]
    ) -> None:
        """Re-running an unchanged failing file must not pile up DLQ rows."""
        repo = ExtractionRepository()
        await repo.save_extraction(failed_extraction_state)

        again = dict(failed_extraction_state)
        again["error"] = "a different, later error"
        again["retry_count"] = 2
        await repo.save_extraction(again)

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM dead_letter_queue")
        finally:
            await conn.close()
        assert len(rows) == 1
        assert rows[0]["last_error"] == "a different, later error"  # refreshed
        assert rows[0]["retry_count"] == 2

    async def test_changed_content_or_other_file_gets_its_own_row(
        self, tmp_db: str, failed_extraction_state: dict[str, Any]
    ) -> None:
        repo = ExtractionRepository()
        await repo.save_extraction(failed_extraction_state)

        edited = dict(failed_extraction_state, raw_content="edited, still broken")
        await repo.save_extraction(edited)

        other_file = dict(failed_extraction_state, file_path="/tmp/other.txt")
        await repo.save_extraction(other_file)

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM dead_letter_queue")
        finally:
            await conn.close()
        assert len(rows) == 3

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


class TestRagSwitchedOff:
    async def test_validated_result_is_not_embedded_when_rag_is_off(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        from src.config import settings
        from src.rag.chroma_client import get_or_create_collection

        assert settings.rag_enabled is False  # the default
        await ExtractionRepository().save_extraction(passed_extraction_state)

        assert get_or_create_collection("M&A").count() == 0
        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT id FROM extractions")
        finally:
            await conn.close()
        assert len(rows) == 1  # the gold row is still written


class TestDLQFailureColumns:
    """The DLQ says where and why a document stopped, and keeps its history."""

    @pytest.fixture
    def classified_failure_state(
        self, failed_extraction_state: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            **failed_extraction_state,
            "doc_type": None,
            "classification": {
                "label": "M&A",
                "evidence_quote": "made up quote",
                "confidence": 0.9,
                "reasoning": "r",
            },
            "failure": {
                "stage": "classify_validate",
                "code": "EVIDENCE_QUOTE_NOT_FOUND",
                "category": "reference",
                "message": "exhausted",
                "details": {"failures": [{"code": "EVIDENCE_QUOTE_NOT_FOUND"}]},
            },
            "correction_logs": [
                {
                    "cycle": 1,
                    "error_summary": "[REFERENCE] quote not found",
                    "raw_feedback": "...",
                    "loop": "classify",
                }
            ],
            "error": "classification_validator: max retries (2) exhausted.",
        }

    async def _only_row(self) -> Any:
        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall("SELECT * FROM dead_letter_queue")
        finally:
            await conn.close()
        assert len(rows) == 1
        return rows[0]

    async def test_stage_code_and_category_are_columns(
        self, tmp_db: str, classified_failure_state: dict[str, Any]
    ) -> None:
        await ExtractionRepository().save_extraction(classified_failure_state)
        row = await self._only_row()

        assert row["failed_stage"] == "classify_validate"
        assert row["error_code"] == "EVIDENCE_QUOTE_NOT_FOUND"
        assert row["error_category"] == "reference"
        assert json.loads(row["failure_details"])["failures"][0]["code"] == (
            "EVIDENCE_QUOTE_NOT_FOUND"
        )

    async def test_correction_history_and_classification_survive(
        self, tmp_db: str, classified_failure_state: dict[str, Any]
    ) -> None:
        """These are otherwise lost: failed docs never reach correction_logs."""
        await ExtractionRepository().save_extraction(classified_failure_state)
        row = await self._only_row()

        logs = json.loads(row["correction_logs"])
        assert logs[0]["loop"] == "classify"
        assert "quote not found" in logs[0]["error_summary"]
        assert json.loads(row["classification"])["evidence_quote"] == "made up quote"

    async def test_error_without_failure_record_gets_a_fallback(
        self, tmp_db: str, failed_extraction_state: dict[str, Any]
    ) -> None:
        await ExtractionRepository().save_extraction(failed_extraction_state)
        row = await self._only_row()

        assert row["failed_stage"] == "pipeline"
        assert row["error_code"] == "UNCLASSIFIED_ERROR"
        assert row["error_category"] == "unknown"
        assert row["classification"] is None

    async def test_repeat_failure_refreshes_the_new_columns(
        self, tmp_db: str, classified_failure_state: dict[str, Any]
    ) -> None:
        repo = ExtractionRepository()
        await repo.save_extraction(classified_failure_state)

        later = dict(classified_failure_state)
        later["failure"] = {
            "stage": "validate",
            "code": "TARGET.NON_EMPTY",
            "category": "missing_info",
            "message": "m",
            "details": None,
        }
        await repo.save_extraction(later)
        row = await self._only_row()

        assert row["failed_stage"] == "validate"
        assert row["error_code"] == "TARGET.NON_EMPTY"
        assert row["failure_details"] is None

    async def test_successful_extraction_logs_keep_their_loop(
        self, tmp_db: str, passed_extraction_state: dict[str, Any]
    ) -> None:
        state = dict(passed_extraction_state)
        state["extracted_data"] = state.get("extracted_data") or {
            "doc_id": "doc-1",
            "acquirer": "A",
            "target": "B",
            "raw_text_snippet": "x",
        }
        state["correction_logs"] = [
            {"cycle": 1, "error_summary": "c", "raw_feedback": "c", "loop": "classify"},
            {"cycle": 1, "error_summary": "e", "raw_feedback": "e", "loop": "extract"},
        ]
        await ExtractionRepository().save_extraction(state)

        conn = await get_connection()
        try:
            rows = await conn.execute_fetchall(
                "SELECT loop, cycle FROM correction_logs ORDER BY id"
            )
        finally:
            await conn.close()
        assert [(r["loop"], r["cycle"]) for r in rows] == [
            ("classify", 1),
            ("extract", 1),
        ]


class TestSchemaMigration:
    async def test_old_database_gains_the_new_columns_without_losing_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import sqlite3

        from src.config import settings

        db_path = str(tmp_path / "old.db")
        old = sqlite3.connect(db_path)
        old.executescript(
            """
            CREATE TABLE dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL, source_hash TEXT NOT NULL, doc_type TEXT,
                last_error TEXT NOT NULL, retry_count INTEGER NOT NULL,
                failed_at TEXT NOT NULL
            );
            CREATE TABLE correction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL,
                cycle INTEGER NOT NULL, error_summary TEXT NOT NULL,
                llm_raw_response TEXT, created_at TEXT NOT NULL
            );
            INSERT INTO dead_letter_queue
                (file_path, source_hash, doc_type, last_error, retry_count, failed_at)
                VALUES ('/old.txt', 'h', 'M&A', 'old error', 3, 'then');
            """
        )
        old.commit()
        old.close()

        monkeypatch.setattr(settings, "sqlite_db_path", db_path)
        await init_db()
        await init_db()  # idempotent

        conn = await get_connection()
        try:
            dlq_cols = {
                r["name"]
                for r in await conn.execute_fetchall(
                    "PRAGMA table_info(dead_letter_queue)"
                )
            }
            log_cols = {
                r["name"]
                for r in await conn.execute_fetchall(
                    "PRAGMA table_info(correction_logs)"
                )
            }
            rows = await conn.execute_fetchall("SELECT * FROM dead_letter_queue")
        finally:
            await conn.close()

        assert {
            "failed_stage",
            "error_code",
            "error_category",
            "failure_details",
            "classification",
            "correction_logs",
        } <= dlq_cols
        assert "loop" in log_cols
        assert "run_id" in dlq_cols
        assert len(rows) == 1
        assert rows[0]["last_error"] == "old error"
        assert rows[0]["failed_stage"] is None  # legacy rows stay NULL
