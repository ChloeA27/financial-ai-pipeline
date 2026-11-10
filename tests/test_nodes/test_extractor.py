"""
Tests for the Extractor Node's handling of LLM output that fails schema
parsing — it must feed the self-correction loop instead of aborting.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.nodes import extractor as extractor_module
from src.nodes.extractor import extractor_node


class FakeOpenAI:
    """Stand-in for ``AsyncOpenAI`` that replays queued raw completions."""

    queue: list[str] = []
    calls: list[list[dict[str, str]]] = []

    def __init__(self, **_: Any) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    async def _create(self, *, messages: list[dict[str, str]], **_: Any) -> Any:
        FakeOpenAI.calls.append(messages)
        content = FakeOpenAI.queue.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAI]:
    FakeOpenAI.queue = []
    FakeOpenAI.calls = []
    monkeypatch.setattr("src.llm.client.AsyncOpenAI", FakeOpenAI)

    async def _no_rag(**_: Any) -> str:
        return ""

    monkeypatch.setattr(extractor_module, "retrieve_context", _no_rag)
    return FakeOpenAI


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "file_path": "/fake/ma.txt",
        "raw_content": "Acme to acquire Widgets Inc. for $1B in cash.",
        "doc_type": "M&A",
        "extracted_data": None,
        "validation_passed": None,
        "validation_report": None,
        "correction_logs": [],
        "retry_count": 0,
        "max_retries": 3,
        "error": None,
    }
    state.update(overrides)
    return state


GOOD = json.dumps(
    {"acquirer": "Acme", "target": "Widgets Inc.", "payment_method": "Cash"}
)


class TestSchemaParseFailure:
    async def test_bad_enum_is_logged_not_fatal(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = [json.dumps({"acquirer": "A", "payment_method": "cash"})]

        update = await extractor_node(_state())

        assert "error" not in update
        assert update["extracted_data"] is None
        assert update["validation_passed"] is False
        assert update["retry_count"] == 1
        [log] = update["correction_logs"]
        assert log["cycle"] == 1
        assert "payment_method" in log["error_summary"]
        assert "[SCHEMA]" in log["error_summary"]

    @pytest.mark.parametrize(
        "payload, field",
        [
            ({"stake_percentage": 150}, "stake_percentage"),
            ({"announcement_date": "01/18/2022"}, "announcement_date"),
            ({"total_value_usd": -5}, "total_value_usd"),
        ],
    )
    async def test_other_schema_violations(
        self, fake_llm: type[FakeOpenAI], payload: dict, field: str
    ):
        fake_llm.queue = [json.dumps(payload)]
        update = await extractor_node(_state())
        assert update["validation_passed"] is False
        assert field in update["correction_logs"][0]["error_summary"]

    async def test_malformed_json_is_logged(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = ["Sorry, I cannot help with that."]
        update = await extractor_node(_state())
        assert update["validation_passed"] is False
        assert update["retry_count"] == 1
        assert "error" not in update

    async def test_last_allowed_retry_is_not_fatal(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = [json.dumps({"payment_method": "cash"})]
        prior = [
            {"cycle": i, "error_summary": "x", "raw_feedback": "x"} for i in (1, 2)
        ]
        update = await extractor_node(
            _state(retry_count=2, max_retries=3, correction_logs=prior)
        )
        assert update["retry_count"] == 3
        assert "error" not in update

    async def test_exhausting_retries_sets_error(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = [json.dumps({"payment_method": "cash"})]
        prior = [
            {"cycle": i, "error_summary": "x", "raw_feedback": "x"} for i in (1, 2, 3)
        ]
        update = await extractor_node(
            _state(retry_count=3, max_retries=3, correction_logs=prior)
        )
        assert update["retry_count"] == 3  # clamped: no 4th retry was performed
        assert [l["cycle"] for l in update["correction_logs"]] == [1, 2, 3, 4]
        assert "max retries (3) exhausted" in update["error"]

    async def test_does_not_mutate_input_logs(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = [json.dumps({"payment_method": "cash"})]
        logs: list = []
        await extractor_node(_state(correction_logs=logs))
        assert logs == []

    async def test_feedback_reaches_next_prompt(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = [json.dumps({"payment_method": "cash"}), GOOD]
        first = await extractor_node(_state())

        retry_state = _state(
            retry_count=first["retry_count"],
            correction_logs=first["correction_logs"],
            validation_passed=False,
        )
        second = await extractor_node(retry_state)

        user_msg = fake_llm.calls[-1][-1]["content"]
        assert "PREVIOUS VALIDATION FEEDBACK" in user_msg
        assert "payment_method" in user_msg
        assert second["extracted_data"]["acquirer"] == "Acme"


class TestSuccessAndOtherErrors:
    async def test_success_resets_validation_flag(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = [GOOD]
        update = await extractor_node(_state(validation_passed=False, retry_count=1))
        assert update["validation_passed"] is None
        assert update["extracted_data"]["target"] == "Widgets Inc."
        assert "error" not in update

    async def test_transport_failure_stays_fatal(
        self, monkeypatch: pytest.MonkeyPatch, fake_llm: type[FakeOpenAI]
    ):
        # Empty queue → IndexError inside the fake client: not a schema error.
        fake_llm.queue = []
        update = await extractor_node(_state())
        assert "LLM call failed" in update["error"]
        assert "retry_count" not in update


class TestFailureRecords:
    """Fatal extractor exits carry a structured failure for the DLQ."""

    async def test_transport_failure_is_transient(self, fake_llm: type[FakeOpenAI]):
        fake_llm.queue = []
        update = await extractor_node(_state())
        assert update["failure"]["stage"] == "extract"
        assert update["failure"]["code"] == "LLM_CALL_FAILED"
        assert update["failure"]["category"] == "transient"

    async def test_exhausted_schema_parsing_is_a_structure_failure(
        self, fake_llm: type[FakeOpenAI]
    ):
        fake_llm.queue = [json.dumps({"payment_method": "cash"})]
        prior = [
            {"cycle": i, "error_summary": "x", "raw_feedback": "x"} for i in (1, 2, 3)
        ]
        update = await extractor_node(
            _state(retry_count=3, max_retries=3, correction_logs=prior)
        )
        assert update["failure"]["stage"] == "extract"
        assert update["failure"]["code"] == "SCHEMA_PARSE_FAILED"
        assert update["failure"]["category"] == "structure"

    async def test_mid_loop_parse_failure_has_no_failure_record(
        self, fake_llm: type[FakeOpenAI]
    ):
        fake_llm.queue = [json.dumps({"payment_method": "cash"})]
        update = await extractor_node(_state())
        assert "failure" not in update

    @pytest.mark.parametrize("doc_type", ["none", "unclear", "Management_Change"])
    async def test_type_without_extractor_is_out_of_scope(
        self, fake_llm: type[FakeOpenAI], doc_type: str
    ):
        update = await extractor_node(_state(doc_type=doc_type))
        assert update["failure"]["code"] == "UNSUPPORTED_DOC_TYPE"
        assert update["failure"]["category"] == "out_of_scope"
        assert fake_llm.calls == []

    async def test_empty_document(self, fake_llm: type[FakeOpenAI]):
        update = await extractor_node(_state(raw_content=" "))
        assert update["failure"]["code"] == "EMPTY_DOCUMENT"
        assert update["failure"]["category"] == "input_error"


class TestRagSwitch:
    async def test_retriever_is_not_called_when_rag_is_off(
        self, fake_llm: type[FakeOpenAI], monkeypatch: pytest.MonkeyPatch
    ):
        calls: list = []

        async def spy(**kw: Any) -> str:
            calls.append(kw)
            return "SHOULD NOT APPEAR"

        monkeypatch.setattr(extractor_module, "retrieve_context", spy)
        fake_llm.queue = [GOOD]
        update = await extractor_node(_state())

        assert calls == []
        assert "SHOULD NOT APPEAR" not in fake_llm.calls[-1][-1]["content"]
        assert update["step_runs"][0]["output"]["rag_context_used"] is False

    async def test_retriever_is_used_when_rag_is_on(
        self, fake_llm: type[FakeOpenAI], monkeypatch: pytest.MonkeyPatch, rag_on: None
    ):
        async def ctx(**_: Any) -> str:
            return "HISTORICAL EXAMPLE"

        monkeypatch.setattr(extractor_module, "retrieve_context", ctx)
        fake_llm.queue = [GOOD]
        update = await extractor_node(_state())

        assert "HISTORICAL EXAMPLE" in fake_llm.calls[-1][-1]["content"]
        assert update["step_runs"][0]["output"]["rag_context_used"] is True
