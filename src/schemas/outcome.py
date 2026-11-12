"""
Where a document ends up after one pipeline run — the single routing rule.

Every run ends in exactly one of four outcomes::

    stored        validated extraction → gold ``extractions``
    out_of_scope  the classifier said ``none``: not a type this pipeline handles.
                  Not a failure, nobody needs to act; kept in bronze/silver only
    human_review  a person must decide about the document's content
    dlq           an engineer must act: infrastructure, input or a pipeline bug

The rule looks only at the structured ``failure.category``; model confidence is
never a routing signal (design notes D10).  Storage and evaluation both call
``outcome_of`` so the two can never disagree about where a document went.

Human review covers the classifier's ``unclear`` label and the content checks
that more retries cannot settle from one failure alone: a quote that is never
found in the source (``reference``), a required value that is absent
(``missing_info``) and a cross-field rule that is violated (``business_rule``).
A model that keeps breaking the output format (``structure``) is an
engineering problem and goes to the DLQ.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.schemas.failure import (
    CAT_AMBIGUOUS,
    CAT_BUSINESS_RULE,
    CAT_MISSING_INFO,
    CAT_OUT_OF_SCOPE,
    CAT_REFERENCE,
)

OUTCOME_STORED = "stored"
OUTCOME_OUT_OF_SCOPE = "out_of_scope"
OUTCOME_HUMAN_REVIEW = "human_review"
OUTCOME_DLQ = "dlq"

OUTCOMES = (OUTCOME_STORED, OUTCOME_OUT_OF_SCOPE, OUTCOME_HUMAN_REVIEW, OUTCOME_DLQ)

HUMAN_REVIEW_CATEGORIES = frozenset(
    {CAT_AMBIGUOUS, CAT_REFERENCE, CAT_MISSING_INFO, CAT_BUSINESS_RULE}
)


def outcome_of(state: Mapping[str, Any]) -> str:
    """Final outcome of one pipeline run (see module docstring)."""
    if state.get("validation_passed") is True and not state.get("error"):
        return OUTCOME_STORED
    category = (state.get("failure") or {}).get("category")
    if category == CAT_OUT_OF_SCOPE:
        return OUTCOME_OUT_OF_SCOPE
    if category in HUMAN_REVIEW_CATEGORIES:
        return OUTCOME_HUMAN_REVIEW
    # structure, transient, input_error, unknown, or no failure record at all
    return OUTCOME_DLQ
