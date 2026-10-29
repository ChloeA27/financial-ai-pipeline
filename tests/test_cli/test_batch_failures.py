"""
Batch mode: a pipeline that raises must still land in the DLQ, and one bad
write must not abort the rest of the batch.
"""

from __future__ import annotations

from typing import Any

import pytest

from src import main as main_module
from src.storage.db import get_connection, init_db
from src.storage.repository import ExtractionRepository


@pytest.fixture
async def tmp_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    from src.config import settings

    db = str(tmp_path / "t.db")
    monkeypatch.setattr(settings, "sqlite_db_path", db)
    await init_db()
    return db


def _states(*names: str) -> list[dict]:
    return [
        {"file_path": f"/data/{n}.txt", "raw_content": f"text of {n}"} for n in names
    ]


GOOD = {
    "validation_passed": True,
    "error": None,
    "doc_type": "M&A",
    "retry_count": 0,
}


async def test_crashing_pipeline_is_quarantined(
    tmp_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake(fp: str) -> dict:
        if fp.endswith("boom.txt"):
            raise RuntimeError("kaboom")
        return {**GOOD, "file_path": fp, "raw_content": "ok", "extracted_data": {}}

    monkeypatch.setattr(main_module, "run_pipeline", fake)
    # The good doc would try to embed/index; keep this test about the DLQ.
    monkeypatch.setattr(
        ExtractionRepository, "_upsert_extraction", _noop_upsert, raising=True
    )

    await main_module._process_batch_concurrent(
        _states("boom", "fine"), ExtractionRepository(), concurrency=2
    )

    conn = await get_connection()
    try:
        rows = await conn.execute_fetchall(
            "SELECT file_path, doc_type, last_error, retry_count FROM dead_letter_queue"
        )
    finally:
        await conn.close()
    assert [tuple(r) for r in rows] == [
        ("/data/boom.txt", None, "pipeline raised RuntimeError: kaboom", 0)
    ]


async def _noop_upsert(self: Any, state: dict) -> None:
    return None


async def test_failed_persist_does_not_abort_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(fp: str) -> dict:
        return {**GOOD, "file_path": fp, "raw_content": "x", "extracted_data": {}}

    saved: list[str] = []

    class Repo:
        async def save_extraction(self, state: dict) -> None:
            if state["file_path"].endswith("a.txt"):
                raise OSError("disk full")
            saved.append(state["file_path"])

    monkeypatch.setattr(main_module, "run_pipeline", fake)
    await main_module._process_batch_concurrent(
        _states("a", "b", "c"), Repo(), concurrency=1  # type: ignore[arg-type]
    )
    assert saved == ["/data/b.txt", "/data/c.txt"]


def test_crashed_state_shape() -> None:
    st = main_module._crashed_state("/x.txt", "raw", ValueError("bad"))
    assert st["error"] == "pipeline raised ValueError: bad"
    assert st["raw_content"] == "raw"
    assert st["validation_passed"] is None
    assert st["retry_count"] == 0
