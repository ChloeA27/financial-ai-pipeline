"""Retry bookkeeping shared by the Validator and the Extractor."""

from __future__ import annotations

import pytest

from src.nodes.retry import register_failure


def _state(**kw):
    base = {"retry_count": 0, "max_retries": 3, "correction_logs": []}
    base.update(kw)
    return base


def _fail(state):
    return register_failure(
        state, error_summary="boom", report="report", exhausted_message="EXHAUSTED"
    )


def test_first_failure_spends_a_retry() -> None:
    upd = _fail(_state())
    assert upd["retry_count"] == 1
    assert "error" not in upd
    assert upd["correction_logs"] == [
        {"cycle": 1, "error_summary": "boom", "raw_feedback": "report"}
    ]


@pytest.mark.parametrize("max_retries", [0, 1, 3, 5])
def test_budget_is_max_retries_retries_plus_first_attempt(max_retries: int) -> None:
    state = _state(max_retries=max_retries)
    attempts = 0
    while True:
        attempts += 1
        upd = _fail(state)
        state = {**state, **upd}
        if "error" in upd:
            break
        assert attempts <= max_retries + 1
    assert attempts == max_retries + 1
    assert state["retry_count"] == max_retries
    assert len(state["correction_logs"]) == max_retries + 1
    assert upd["error"] == "EXHAUSTED"


def test_input_logs_not_mutated() -> None:
    logs: list = []
    _fail(_state(correction_logs=logs))
    assert logs == []
