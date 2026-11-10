"""
Audit trail end to end: every node execution leaves a ``step_runs`` record in
the state, and the repository persists them — for failed documents too — next
to the raw (bronze) copy of the source text.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.graph.builder import run_pipeline
from src.storage.db import get_connection, init_db
from src.storage.raw_store import read_raw
from src.storage.repository import ExtractionRepository
from src.utils.hashing import content_hash
from tests.test_nodes.test_extractor import FakeOpenAI

TEXT = "Acme to acquire Widgets Inc. for $1B in cash."

CLASSIFY = (
    '{"doc_type": "M&A", "evidence_quote": "Acme to acquire Widgets Inc. for $1B",'
    ' "confidence": 0.9, "reasoning": "deal"}'
)
BAD_QUOTE = (
    '{"doc_type": "M&A", "evidence_quote": "agreed to buy Widgets outright",'
    ' "confidence": 0.9, "reasoning": "deal"}'
)
NONE = '{"doc_type": "none", "confidence": 0.8, "reasoning": "earnings"}'
GOOD = (
    '{"acquirer": "Acme", "target": "Widgets Inc.", "total_value_usd": 1e9,'
    ' "payment_method": "Cash"}'
)
BAD_ENUM = '{"acquirer": "Acme", "target": "W", "payment_method": "cash"}'
NO_ACQUIRER = '{"target": "W", "total_value_usd": 5, "payment_method": "Cash"}'


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from src.nodes import extractor as extractor_module

    FakeOpenAI.queue = []
    FakeOpenAI.calls = []
    monkeypatch.setattr("src.llm.client.AsyncOpenAI", FakeOpenAI)

    async def _no_rag(**_: Any) -> str:
        return ""

    monkeypatch.setattr(extractor_module, "retrieve_context", _no_rag)
    src = tmp_path / "ma.txt"
    src.write_text(TEXT)
    return FakeOpenAI, str(src)


@pytest.fixture
async def tmp_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    from src.config import settings

    db_path = str(tmp_path / "audit.db")
    monkeypatch.setattr(settings, "sqlite_db_path", db_path)
    await init_db()
    return db_path


def _trail(state: dict) -> list[tuple]:
    return [
        (s["stage"], s["attempt"], s["outcome"], s["error_code"])
        for s in state["step_runs"]
    ]


async def _rows(sql: str, *args: Any) -> list[Any]:
    conn = await get_connection()
    try:
        return await conn.execute_fetchall(sql, args)
    finally:
        await conn.close()


class TestStepRunsInState:
    async def test_every_attempt_of_both_loops_is_recorded(self, scripted) -> None:
        fake, path = scripted
        fake.queue = [BAD_QUOTE, CLASSIFY, BAD_ENUM, NO_ACQUIRER, GOOD]

        result = await run_pipeline(path)

        assert result["validation_passed"] is True
        assert _trail(result) == [
            ("classify", 1, "passed", None),
            ("classify_validate", 1, "rejected", "EVIDENCE_QUOTE_NOT_FOUND"),
            ("classify", 2, "passed", None),
            ("classify_validate", 2, "passed", None),
            ("extract", 1, "rejected", "SCHEMA_PARSE_FAILED"),
            ("extract", 2, "passed", None),
            ("validate", 2, "rejected", "ACQUIRER.NON_EMPTY"),
            ("extract", 3, "passed", None),
            ("validate", 3, "passed", None),
        ]

    async def test_records_share_the_run_and_point_at_the_source(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [CLASSIFY, GOOD]

        result = await run_pipeline(path)

        assert {s["run_id"] for s in result["step_runs"]} == {result["run_id"]}
        assert {s["source_hash"] for s in result["step_runs"]} == {content_hash(TEXT)}
        assert result["source_hash"] == content_hash(TEXT)

    async def test_llm_steps_keep_the_raw_reply_and_validators_do_not_call_llm(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [CLASSIFY, BAD_ENUM, GOOD]

        result = await run_pipeline(path)
        by_stage = {}
        for step in result["step_runs"]:
            by_stage.setdefault(step["stage"], []).append(step)

        assert by_stage["classify"][0]["raw_response"] == CLASSIFY
        # the unparsable reply is kept, even though nothing could be parsed from it
        assert by_stage["extract"][0]["raw_response"] == BAD_ENUM
        assert "payment_method" in by_stage["extract"][0]["output"]["schema_errors"]
        for stage in ("classify", "extract"):
            assert all(
                s["prompt_version"].startswith("sha256:") for s in by_stage[stage]
            )
            assert all(s["model_name"] for s in by_stage[stage])
        for stage in ("classify_validate", "validate"):
            assert all(s["raw_response"] is None for s in by_stage[stage])
            assert all(s["model_name"] is None for s in by_stage[stage])

    async def test_extraction_output_is_recorded_for_rejected_attempts(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [CLASSIFY, NO_ACQUIRER, GOOD]

        result = await run_pipeline(path)

        first_extract = [s for s in result["step_runs"] if s["stage"] == "extract"][0]
        assert first_extract["output"]["fields"]["target"] == "W"
        assert first_extract["output"]["fields"]["acquirer"] is None
        assert first_extract["output"]["rag_context_used"] is False

    async def test_terminal_label_is_one_fatal_check(self, scripted) -> None:
        fake, path = scripted
        fake.queue = [NONE]

        result = await run_pipeline(path)

        assert _trail(result) == [
            ("classify", 1, "passed", None),
            ("classify_validate", 1, "fatal", "UNSUPPORTED_DOC_TYPE"),
        ]
        assert result["step_runs"][-1]["error_category"] == "out_of_scope"

    async def test_two_runs_of_the_same_file_get_different_run_ids(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [NONE, NONE]
        first = await run_pipeline(path)
        second = await run_pipeline(path)
        assert first["run_id"] != second["run_id"]
        assert first["source_hash"] == second["source_hash"]


class TestPersistedAuditTrail:
    async def test_failed_document_keeps_raw_text_and_every_step(
        self, scripted, tmp_db: str
    ) -> None:
        fake, path = scripted
        fake.queue = [CLASSIFY] + [NO_ACQUIRER] * 4

        state = await run_pipeline(path)
        await ExtractionRepository().save_extraction(state)

        [doc] = await _rows("SELECT * FROM documents")
        assert doc["last_status"] == "dlq"
        assert await read_raw(doc["raw_uri"]) == TEXT

        steps = await _rows(
            "SELECT stage, attempt, outcome, output, run_id FROM step_runs ORDER BY id"
        )
        assert len(steps) == 2 + 4 * 2  # classify pair + 4 extract/validate pairs
        assert {s["run_id"] for s in steps} == {state["run_id"]}
        assert steps[-1]["outcome"] == "fatal"
        rejected_extract = json.loads(steps[2]["output"])
        assert rejected_extract["fields"]["target"] == "W"

        [dlq] = await _rows("SELECT run_id, failed_stage FROM dead_letter_queue")
        assert dlq["run_id"] == state["run_id"]
        assert dlq["failed_stage"] == "validate"

    async def test_passed_document_links_gold_row_to_run_and_source(
        self, scripted, tmp_db: str
    ) -> None:
        fake, path = scripted
        fake.queue = [CLASSIFY, GOOD]

        state = await run_pipeline(path)
        await ExtractionRepository().save_extraction(state)

        [row] = await _rows(
            "SELECT run_id, source_hash, extracted_data FROM extractions"
        )
        assert row["run_id"] == state["run_id"]
        assert row["source_hash"] == content_hash(TEXT)
        data = json.loads(row["extracted_data"])
        assert data["metadata"]["source_hash"] == content_hash(TEXT)
        assert "raw_text_snippet" not in data

        [doc] = await _rows("SELECT last_status FROM documents")
        assert doc["last_status"] == "stored"
        assert len(await _rows("SELECT id FROM step_runs")) == 4

    async def test_rerun_registers_the_document_once_and_appends_steps(
        self, scripted, tmp_db: str
    ) -> None:
        fake, path = scripted
        repo = ExtractionRepository()
        fake.queue = [NONE, NONE]
        first = await run_pipeline(path)
        await repo.save_extraction(first)
        second = await run_pipeline(path)
        await repo.save_extraction(second)

        [doc] = await _rows("SELECT last_run_id FROM documents")
        assert doc["last_run_id"] == second["run_id"]
        runs = await _rows("SELECT DISTINCT run_id FROM step_runs")
        assert {r["run_id"] for r in runs} == {first["run_id"], second["run_id"]}
