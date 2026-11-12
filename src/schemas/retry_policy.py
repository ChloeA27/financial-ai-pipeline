"""
Which failures may be retried — the retry policy, by failure category.

Retrying is worth it only when another attempt can fix the problem without
pushing the model to invent something:

    structure    retry with feedback (bad shape; the model usually fixes it)
    reference    retry with feedback (the quote is checked against the source,
                 so a retry cannot pass by inventing one)
    transient    retried by the LLM client (SDK backoff, ``llm_max_retries``),
                 never by the self-correction loops

Never retried:

    missing_info   feedback "field X is missing" pushes the model to fill it.
                   Measured: scenario H2 (results/scenarios/20261112-211858)
                   wrote a description into an unnamed target after one retry
    business_rule  the feedback states the constraint, not the truth, so a retry
                   can satisfy it by changing any date or number
    config_error, input_error, internal_error, unknown, ambiguous, out_of_scope
                   another attempt cannot change the answer

How many retries a retryable failure gets is the loop's budget
(``classify_max_retries``, ``max_retries``).
"""

from __future__ import annotations

from src.schemas.failure import CAT_REFERENCE, CAT_STRUCTURE

RETRYABLE_CATEGORIES = frozenset({CAT_STRUCTURE, CAT_REFERENCE})


def is_retryable(category: str) -> bool:
    """True when a self-correction retry may fix a failure of this category."""
    return category in RETRYABLE_CATEGORIES
