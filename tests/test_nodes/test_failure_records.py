"""
Structured failure records: every fatal exit says which stage stopped, a stable
code, and a category — and mid-loop retries do NOT write a failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.nodes.reader import reader_node
from src.nodes.retry import register_classification_failure, register_failure
from src.nodes.extraction_validator import extraction_validator_node
from src.schemas.failure import fallback_failure, make_failure


def _exhausted(state: dict[str, Any]) -> dict[str, Any]:
    """The same state with the extraction budget already spent."""
    return {**state, "retry_count": state["max_retries"]}


class TestValidatorFailureAttribution:
    async def test_mid_loop_failure_has_no_failure_record(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        sample_pipeline_state["extracted_data"]["target"] = ""
        update = await extraction_validator_node(sample_pipeline_state)
        assert update["validation_passed"] is False
        assert "failure" not in update  # only terminal exits carry one
        assert "error" not in update

    async def test_missing_required_field(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        sample_pipeline_state["extracted_data"]["target"] = None
        update = await extraction_validator_node(_exhausted(sample_pipeline_state))

        failure = update["failure"]
        assert failure["stage"] == "validate"
        assert failure["code"] == "TARGET.NON_EMPTY"
        assert failure["category"] == "missing_info"

    async def test_present_but_not_allowed_value_is_structure(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        sample_pipeline_state["extracted_data"]["payment_method"] = "bitcoin"
        update = await extraction_validator_node(_exhausted(sample_pipeline_state))

        assert update["failure"]["code"] == "PAYMENT_METHOD.ALLOWED_VALUES"
        assert update["failure"]["category"] == "structure"

    async def test_cross_field_rule_is_a_business_rule(
        self, dividend_pipeline_state: dict[str, Any]
    ) -> None:
        # Payment before declaration: breaks the timeline invariant.
        dividend_pipeline_state["extracted_data"]["payment_date"] = "2026-01-01"
        update = await extraction_validator_node(_exhausted(dividend_pipeline_state))

        assert update["failure"]["code"] == "TIMELINE_ORDER"
        assert update["failure"]["category"] == "business_rule"

    async def test_most_severe_failure_is_primary_and_all_are_listed(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        data = sample_pipeline_state["extracted_data"]
        data["target"] = None  # CRITICAL
        data["payment_method"] = "bitcoin"  # LOW
        update = await extraction_validator_node(_exhausted(sample_pipeline_state))

        failure = update["failure"]
        assert failure["code"] == "TARGET.NON_EMPTY"
        codes = {r["code"] for r in failure["details"]["failed_rules"]}
        assert codes == {"TARGET.NON_EMPTY", "PAYMENT_METHOD.ALLOWED_VALUES"}

    async def test_no_extracted_data(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        sample_pipeline_state["extracted_data"] = None
        update = await extraction_validator_node(sample_pipeline_state)
        assert update["failure"]["code"] == "NO_EXTRACTED_DATA"
        assert update["failure"]["stage"] == "validate"

    async def test_passing_document_has_no_failure(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        update = await extraction_validator_node(sample_pipeline_state)
        assert update["validation_passed"] is True
        assert "failure" not in update


class TestReaderFailures:
    async def test_missing_file(self, tmp_path) -> None:
        update = await reader_node(
            {"file_path": str(tmp_path / "nope.txt"), "raw_content": ""}
        )
        assert update["failure"]["stage"] == "read"
        assert update["failure"]["code"] == "FILE_NOT_FOUND"
        assert update["failure"]["category"] == "input_error"

    async def test_no_file_path(self) -> None:
        update = await reader_node({"file_path": "", "raw_content": ""})
        assert update["failure"]["code"] == "NO_FILE_PATH"


class TestRetryLoopsAreIndependent:
    def _state(self, **kw: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "retry_count": 0,
            "max_retries": 3,
            "classify_retry_count": 0,
            "classify_max_retries": 2,
            "correction_logs": [],
        }
        base.update(kw)
        return base

    def _classify_fail(self, state: dict[str, Any]) -> dict[str, Any]:
        return register_classification_failure(
            state,
            error_summary="c",
            report="c",
            exhausted_message="CLASSIFY EXHAUSTED",
            failure=make_failure("classify", "X", "structure", "m"),
        )

    def _extract_fail(self, state: dict[str, Any]) -> dict[str, Any]:
        return register_failure(
            state, error_summary="e", report="e", exhausted_message="EXTRACT EXHAUSTED"
        )

    def test_classify_budget_is_classify_max_retries_plus_one(self) -> None:
        state = self._state()
        attempts = 0
        while True:
            attempts += 1
            update = self._classify_fail(state)
            state = {**state, **update}
            if "error" in update:
                break
        assert attempts == 3  # 1 first attempt + 2 retries
        assert state["classify_retry_count"] == 2
        assert state["retry_count"] == 0  # extraction budget untouched
        assert state["failure"]["code"] == "X"

    def test_exhausting_one_loop_leaves_the_other_budget_intact(self) -> None:
        state = self._state(classify_retry_count=2)
        state = {**state, **self._classify_fail(state)}
        assert "error" in state
        update = self._extract_fail({**state, "error": None})
        assert update["retry_count"] == 1
        assert "error" not in update

    def test_cycles_are_numbered_per_loop(self) -> None:
        state = self._state()
        state = {**state, **self._classify_fail(state)}
        state = {**state, **self._extract_fail(state)}
        state = {**state, **self._classify_fail(state)}
        state = {**state, **self._extract_fail(state)}
        by_loop = {
            loop: [l["cycle"] for l in state["correction_logs"] if l["loop"] == loop]
            for loop in ("classify", "extract")
        }
        assert by_loop == {"classify": [1, 2], "extract": [1, 2]}

    def test_mid_loop_failure_does_not_write_a_failure_record(self) -> None:
        update = self._classify_fail(self._state())
        assert "failure" not in update
        assert "error" not in update


class TestFallbackFailure:
    def test_unknown_errors_still_get_stage_and_code(self) -> None:
        failure = fallback_failure("something odd")
        assert failure["stage"] == "pipeline"
        assert failure["code"] == "UNCLASSIFIED_ERROR"
        assert failure["category"] == "unknown"
        assert failure["message"] == "something odd"

    @pytest.mark.parametrize("error", [None, ""])
    def test_missing_message_gets_a_default(self, error) -> None:
        assert fallback_failure(error)["message"] == "Unknown error"
