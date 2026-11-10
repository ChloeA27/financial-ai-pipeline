"""
Ingestion is byte-exact: the raw layer keeps the file's bytes unchanged, the
hash matches ``sha256sum`` of the file, and text the pipeline cannot decode
lands in the DLQ instead of being silently skipped.
"""

from __future__ import annotations

import codecs
import hashlib
from pathlib import Path
from typing import Any

import pytest

from src.nodes.reader import (
    SourceDecodeError,
    decode_source,
    read_directory,
    read_single_file,
)
from src.storage.raw_store import read_raw_bytes

# 0x81 is undefined in Windows-1252 and invalid as UTF-8 here → undecodable
UNDECODABLE = b"Deal value \x81\x8d unknown"


class TestDecodeSource:
    def test_plain_utf8(self) -> None:
        assert decode_source("Café merger".encode()) == ("Café merger", "utf-8")

    def test_bom_is_stripped_and_recorded(self) -> None:
        text, label = decode_source(codecs.BOM_UTF8 + b"hello")
        assert (text, label) == ("hello", "utf-8-bom")

    def test_windows_1252_smart_quotes_fall_back(self) -> None:
        text, label = decode_source(b"\x93Definitive agreement\x94 \x96 signed")
        assert label == "cp1252"
        assert text == "“Definitive agreement” – signed"

    @pytest.mark.parametrize(
        "raw, expected",
        [(b"a\r\nb\r\n", "a\nb\n"), (b"a\rb", "a\nb"), (b"a\nb", "a\nb")],
    )
    def test_line_endings_are_normalised_for_the_llm(
        self, raw: bytes, expected: str
    ) -> None:
        assert decode_source(raw)[0] == expected

    def test_binary_content_is_rejected(self) -> None:
        with pytest.raises(SourceDecodeError, match="binary"):
            decode_source(b"%PDF-1.7\x00\x01\x02")

    def test_undecodable_bytes_are_rejected(self) -> None:
        with pytest.raises(SourceDecodeError):
            decode_source(UNDECODABLE)


class TestByteExactIngestion:
    async def test_crlf_file_is_stored_unchanged(self, tmp_path: Path) -> None:
        original = b"Acme to acquire\r\nWidgets Inc.\r\n"
        f = tmp_path / "crlf.txt"
        f.write_bytes(original)

        state = await read_single_file(str(f))

        assert await read_raw_bytes(state["raw_uri"]) == original
        assert state["source_hash"] == hashlib.sha256(original).hexdigest()
        assert state["raw_content"] == "Acme to acquire\nWidgets Inc.\n"
        assert state["source_encoding"] == "utf-8"

    async def test_windows_1252_file_is_processed_and_flagged(
        self, tmp_path: Path
    ) -> None:
        original = b"Acme \x93agrees\x94 to acquire Widgets"
        f = tmp_path / "cp1252.txt"
        f.write_bytes(original)

        state = await read_single_file(str(f))

        assert state.get("error") is None
        assert state["source_encoding"] == "cp1252"
        assert "“agrees”" in state["raw_content"]
        assert await read_raw_bytes(state["raw_uri"]) == original

    async def test_undecodable_file_becomes_a_read_failure(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "bad.txt"
        f.write_bytes(UNDECODABLE)

        state = await read_single_file(str(f))

        assert "cannot decode bad.txt" in state["error"]
        failure = state["failure"]
        assert (failure["stage"], failure["code"], failure["category"]) == (
            "read",
            "DECODE_FAILED",
            "input_error",
        )
        assert failure["details"]["size_bytes"] == len(UNDECODABLE)
        assert state["raw_content"] == ""
        assert state["source_encoding"] is None
        # the bytes are kept even though they could not be decoded
        assert await read_raw_bytes(state["raw_uri"]) == UNDECODABLE

    async def test_batch_listing_keeps_undecodable_files(self, tmp_path: Path) -> None:
        (tmp_path / "good.txt").write_text("Acme to acquire Widgets.")
        (tmp_path / "bad.txt").write_bytes(UNDECODABLE)

        states = await read_directory(str(tmp_path))

        by_name = {Path(s["file_path"]).name: s for s in states}
        assert set(by_name) == {"good.txt", "bad.txt"}  # nothing silently dropped
        assert by_name["bad.txt"]["failure"]["code"] == "DECODE_FAILED"


class TestUndecodableFileEndToEnd:
    async def test_lands_in_the_dlq_without_any_llm_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import settings
        from src.graph.builder import run_pipeline
        from src.storage.db import get_connection, init_db
        from src.storage.repository import ExtractionRepository
        from tests.test_nodes.test_extractor import FakeOpenAI

        FakeOpenAI.queue, FakeOpenAI.calls = [], []
        monkeypatch.setattr("src.llm.client.AsyncOpenAI", FakeOpenAI)
        monkeypatch.setattr(settings, "sqlite_db_path", str(tmp_path / "e2e.db"))
        await init_db()
        f = tmp_path / "bad.txt"
        f.write_bytes(UNDECODABLE)

        state = await run_pipeline(str(f))
        await ExtractionRepository().save_extraction(state)

        assert FakeOpenAI.calls == []
        conn = await get_connection()
        try:
            [dlq] = await conn.execute_fetchall(
                "SELECT failed_stage, error_code FROM dead_letter_queue"
            )
            [doc] = await conn.execute_fetchall(
                "SELECT size_bytes, encoding, last_status FROM documents"
            )
            steps = await conn.execute_fetchall("SELECT id FROM step_runs")
        finally:
            await conn.close()
        assert (dlq["failed_stage"], dlq["error_code"]) == ("read", "DECODE_FAILED")
        assert (doc["size_bytes"], doc["encoding"], doc["last_status"]) == (
            len(UNDECODABLE),
            None,
            "dlq",
        )
        assert steps == []  # no node did any work

    async def test_documents_row_records_encoding_and_byte_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import settings
        from src.storage.db import get_connection, init_db
        from src.storage.repository import ExtractionRepository

        monkeypatch.setattr(settings, "sqlite_db_path", str(tmp_path / "docs.db"))
        await init_db()
        original = b"Acme \x93agrees\x94\r\n"
        f = tmp_path / "cp.txt"
        f.write_bytes(original)
        state: dict[str, Any] = await read_single_file(str(f))
        state["error"] = "stop here"  # persistence only; no graph run needed

        await ExtractionRepository().save_extraction(state)

        conn = await get_connection()
        try:
            [doc] = await conn.execute_fetchall("SELECT * FROM documents")
        finally:
            await conn.close()
        assert doc["encoding"] == "cp1252"
        assert doc["size_bytes"] == len(original)
        assert doc["source_hash"] == hashlib.sha256(original).hexdigest()
