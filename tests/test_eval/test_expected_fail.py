"""
Golden-eval ``expected_fail`` scoring: only a genuine rejection earns credit.
An infrastructure error (LLM unreachable, ...) must not look like success.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("evaluate_script", _SCRIPT)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)  # type: ignore[union-attr]


def _score(**state) -> tuple[int, bool, str]:
    passed, total, [d] = evaluate._eval_expected_fail({}, state)
    assert total == 1
    return passed, d["passed"], d["detail"]


class TestGenuineRejections:
    def test_validation_failed_after_retries(self) -> None:
        p, ok, _ = _score(
            doc_type="M&A",
            validation_passed=False,
            error="validator_node: max retries (3) exhausted.",
        )
        assert (p, ok) == (1, True)

    def test_validation_failed_without_error_flag(self) -> None:
        assert _score(doc_type="M&A", validation_passed=False)[0] == 1

    def test_schema_parse_loop_exhausted(self) -> None:
        p, _, _ = _score(
            doc_type="Dividend",
            validation_passed=False,
            error="extractor_node: max retries (3) exhausted. LLM output never matched",
        )
        assert p == 1

    @pytest.mark.parametrize("doc_type", ["Unknown", "Management_Change"])
    def test_classified_as_unsupported_type(self, doc_type: str) -> None:
        p, _, detail = _score(
            doc_type=doc_type,
            validation_passed=None,
            error=f"extractor_node: no model registered for doc_type='{doc_type}'",
        )
        assert p == 1
        assert doc_type in detail


class TestInfrastructureFailuresEarnNoCredit:
    def test_classifier_connection_error(self) -> None:
        p, ok, detail = _score(
            doc_type=None,
            validation_passed=None,
            error="classifier_node: LLM call failed — Connection error.",
        )
        assert (p, ok) == (0, False)
        assert "infrastructure" in detail

    def test_extractor_llm_error_after_classification(self) -> None:
        p, _, _ = _score(
            doc_type="M&A",
            validation_passed=None,
            error="extractor_node: LLM call failed — 401 Unauthorized",
        )
        assert p == 0

    def test_reader_error(self) -> None:
        assert _score(doc_type=None, error="reader_node: file missing")[0] == 0

    def test_pipeline_never_reached_validator(self) -> None:
        p, _, detail = _score(doc_type="M&A", validation_passed=None, error=None)
        assert p == 0
        assert "None" in detail


def test_passing_a_should_fail_document_is_a_failure() -> None:
    p, _, detail = _score(doc_type="M&A", validation_passed=True)
    assert p == 0
    assert "expected fail" in detail
