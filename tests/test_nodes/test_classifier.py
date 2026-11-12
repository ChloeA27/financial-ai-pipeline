"""
Classifier Node: enum labels, evidence quote, and the classify self-correction
loop (unparsable output is retried with feedback, not fatal).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.nodes.classifier import classifier_node
from src.schemas.classification import ClassificationResult
from tests.test_nodes.test_extractor import FakeOpenAI

TEXT = "Acme to acquire Widgets Inc. for $1B in cash."


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAI]:
    FakeOpenAI.queue = []
    FakeOpenAI.calls = []
    monkeypatch.setattr("src.llm.client.AsyncOpenAI", FakeOpenAI)
    return FakeOpenAI


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "file_path": "/fake/ma.txt",
        "raw_content": TEXT,
        "doc_type": None,
        "classification": None,
        "classification_passed": None,
        "classify_retry_count": 0,
        "classify_max_retries": 2,
        "correction_logs": [],
        "retry_count": 0,
        "max_retries": 3,
        "error": None,
        "failure": None,
    }
    state.update(overrides)
    return state


def _answer(label: str = "M&A", quote: str = "Acme to acquire Widgets Inc.", **kw: Any):
    payload = {
        "doc_type": label,
        "evidence_quote": quote,
        "confidence": 0.9,
        "reasoning": "deal",
    }
    payload.update(kw)
    return json.dumps(payload)


class TestSchema:
    @pytest.mark.parametrize("label", ["M&A", "Dividend", "none", "unclear"])
    def test_known_labels_parse(self, label: str) -> None:
        parsed = ClassificationResult.model_validate_json(_answer(label))
        assert parsed.doc_type == label

    @pytest.mark.parametrize("label", ["Management_Change", "Unknown", "m&a", "M&A "])
    def test_other_strings_are_rejected(self, label: str) -> None:
        with pytest.raises(Exception):
            ClassificationResult.model_validate_json(_answer(label))

    def test_quote_defaults_to_empty(self) -> None:
        raw = json.dumps({"doc_type": "none", "confidence": 0.5, "reasoning": "x"})
        assert ClassificationResult.model_validate_json(raw).evidence_quote == ""


class TestSuccess:
    async def test_label_quote_and_signals_are_kept(self, fake_llm) -> None:
        fake_llm.queue = [_answer()]
        update = await classifier_node(_state())

        assert update["doc_type"] == "M&A"
        assert update["classification"] == {
            "label": "M&A",
            "evidence_quote": "Acme to acquire Widgets Inc.",
            "confidence": 0.9,
            "reasoning": "deal",
        }
        assert update["classification_passed"] is None
        assert "error" not in update

    @pytest.mark.parametrize("label", ["none", "unclear"])
    async def test_terminal_labels_need_no_quote(self, fake_llm, label: str) -> None:
        fake_llm.queue = [_answer(label, quote="")]
        update = await classifier_node(_state())
        assert update["doc_type"] == label
        assert "error" not in update  # judged by the validator, not here

    async def test_success_resets_a_stale_rejection_flag(self, fake_llm) -> None:
        fake_llm.queue = [_answer()]
        update = await classifier_node(_state(classification_passed=False))
        assert update["classification_passed"] is None


class TestSchemaParseFailure:
    @pytest.mark.parametrize(
        "raw",
        [
            _answer("Management_Change"),
            "Sorry, I cannot help with that.",
            json.dumps({"doc_type": "M&A", "reasoning": "no confidence field"}),
        ],
    )
    async def test_unparsable_output_is_logged_not_fatal(self, fake_llm, raw) -> None:
        fake_llm.queue = [raw]
        update = await classifier_node(_state(doc_type="Dividend"))

        assert "error" not in update
        assert update["classification_passed"] is False
        assert update["classification"] is None
        assert update["doc_type"] is None  # the stale label is cleared
        assert update["classify_retry_count"] == 1
        [log] = update["correction_logs"]
        assert log["loop"] == "classify"
        assert log["cycle"] == 1
        assert "[SCHEMA]" in log["error_summary"]

    async def test_does_not_spend_the_extraction_budget(self, fake_llm) -> None:
        fake_llm.queue = ["not json"]
        update = await classifier_node(_state())
        assert "retry_count" not in update

    async def test_exhausted_budget_is_fatal_with_failure_record(
        self, fake_llm
    ) -> None:
        fake_llm.queue = ["not json"]
        prior = [
            {"cycle": i, "error_summary": "x", "raw_feedback": "x", "loop": "classify"}
            for i in (1, 2)
        ]
        update = await classifier_node(
            _state(classify_retry_count=2, correction_logs=prior)
        )

        assert update["classify_retry_count"] == 2  # clamped
        assert "max retries (2) exhausted" in update["error"]
        failure = update["failure"]
        assert failure["stage"] == "classify"
        assert failure["code"] == "CLASSIFICATION_PARSE_FAILED"
        assert failure["category"] == "structure"
        assert [l["cycle"] for l in update["correction_logs"]] == [1, 2, 3]

    async def test_cycles_are_counted_per_loop(self, fake_llm) -> None:
        fake_llm.queue = ["not json"]
        extract_logs = [
            {"cycle": i, "error_summary": "x", "raw_feedback": "x", "loop": "extract"}
            for i in (1, 2, 3)
        ]
        update = await classifier_node(_state(correction_logs=extract_logs))
        assert update["correction_logs"][-1]["cycle"] == 1


class TestFeedbackAndFailures:
    async def test_feedback_reaches_next_prompt(self, fake_llm) -> None:
        fake_llm.queue = ["not json", _answer()]
        first = await classifier_node(_state())

        second = await classifier_node(
            _state(
                correction_logs=first["correction_logs"],
                classify_retry_count=first["classify_retry_count"],
                classification_passed=False,
            )
        )
        user_msg = fake_llm.calls[-1][-1]["content"]
        assert "PREVIOUS CLASSIFICATION FEEDBACK" in user_msg
        assert "[SCHEMA]" in user_msg
        assert second["doc_type"] == "M&A"

    async def test_first_pass_has_no_feedback_block(self, fake_llm) -> None:
        fake_llm.queue = [_answer()]
        await classifier_node(_state())
        assert (
            "PREVIOUS CLASSIFICATION FEEDBACK" not in fake_llm.calls[-1][-1]["content"]
        )

    async def test_extraction_logs_do_not_leak_into_classifier_feedback(
        self, fake_llm
    ) -> None:
        fake_llm.queue = [_answer()]
        extract_logs = [
            {"cycle": 1, "error_summary": "EXTRACT-ONLY", "raw_feedback": "x"}
        ]
        await classifier_node(_state(correction_logs=extract_logs))
        assert "EXTRACT-ONLY" not in fake_llm.calls[-1][-1]["content"]

    async def test_transport_failure_is_fatal_and_transient(self, fake_llm) -> None:
        fake_llm.queue = []  # IndexError inside the fake client: not a schema error
        update = await classifier_node(_state())
        assert "LLM call failed" in update["error"]
        assert update["failure"]["code"] == "LLM_CALL_FAILED"
        assert update["failure"]["category"] == "transient"
        assert update["failure"]["stage"] == "classify"
        assert "classify_retry_count" not in update

    async def test_empty_document(self, fake_llm) -> None:
        update = await classifier_node(_state(raw_content="   "))
        assert update["failure"]["code"] == "EMPTY_DOCUMENT"
        assert update["failure"]["category"] == "input_error"
        assert fake_llm.calls == []

    async def test_skips_when_a_previous_node_failed(self, fake_llm) -> None:
        assert await classifier_node(_state(error="boom")) == {}
        assert fake_llm.calls == []


class TestThinkingMode:
    """The classifier turns DeepSeek thinking off by default (output truncation fix)."""

    async def test_request_disables_thinking_and_step_records_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        sent: list[dict[str, Any]] = []

        class CapturingOpenAI:
            def __init__(self, **_: Any) -> None:
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            async def _create(self, **kwargs: Any) -> Any:
                sent.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=_answer()))
                    ]
                )

            async def close(self) -> None:
                return None

        monkeypatch.setattr("src.llm.client.AsyncOpenAI", CapturingOpenAI)
        update = await classifier_node(_state())

        assert sent[0]["extra_body"] == {"thinking": {"type": "disabled"}}
        assert sent[0]["max_tokens"] == 1024
        recorded = update["step_runs"][0]["output"]["input"]
        assert recorded["thinking"] == "disabled"
        assert recorded["max_tokens"] == 1024
        assert "reasoning_tokens" in recorded
