"""outcome_of: one routing rule, driven by the structured failure category."""

from __future__ import annotations

import pytest

from src.schemas.failure import (
    CAT_AMBIGUOUS,
    CAT_BUSINESS_RULE,
    CAT_INPUT_ERROR,
    CAT_MISSING_INFO,
    CAT_OUT_OF_SCOPE,
    CAT_REFERENCE,
    CAT_STRUCTURE,
    CAT_TRANSIENT,
    CAT_UNKNOWN,
)
from src.schemas.outcome import (
    OUTCOME_DLQ,
    OUTCOME_HUMAN_REVIEW,
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_STORED,
    outcome_of,
)


def _failed(category: str) -> dict:
    return {
        "validation_passed": None,
        "error": "stopped",
        "failure": {"stage": "x", "code": "X", "category": category},
    }


def test_validated_without_error_is_stored() -> None:
    assert outcome_of({"validation_passed": True, "error": None}) == OUTCOME_STORED


def test_an_error_overrides_a_passed_validation() -> None:
    assert outcome_of({"validation_passed": True, "error": "boom"}) == OUTCOME_DLQ


def test_none_label_is_out_of_scope_not_a_failure_queue() -> None:
    assert outcome_of(_failed(CAT_OUT_OF_SCOPE)) == OUTCOME_OUT_OF_SCOPE


@pytest.mark.parametrize(
    "category", [CAT_AMBIGUOUS, CAT_REFERENCE, CAT_MISSING_INFO, CAT_BUSINESS_RULE]
)
def test_content_questions_go_to_a_human(category: str) -> None:
    assert outcome_of(_failed(category)) == OUTCOME_HUMAN_REVIEW


@pytest.mark.parametrize(
    "category", [CAT_STRUCTURE, CAT_TRANSIENT, CAT_INPUT_ERROR, CAT_UNKNOWN]
)
def test_engineering_problems_go_to_the_dlq(category: str) -> None:
    assert outcome_of(_failed(category)) == OUTCOME_DLQ


def test_failure_without_a_record_goes_to_the_dlq() -> None:
    assert outcome_of({"validation_passed": False, "error": "x"}) == OUTCOME_DLQ
    assert outcome_of({}) == OUTCOME_DLQ
