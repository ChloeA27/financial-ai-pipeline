"""
Classification Validator Node — deterministic checks on the Classifier's output.

Plain code, no LLM.  It sits between the Classifier and the Extractor so a bad
classification is caught *before* an expensive and misleading extraction, and
so failures can be attributed to the classification stage.

Checks (on a label that has an extraction path, i.e. M&A / Dividend):
  1. structure  — the label is one of the known labels
  2. reference  — ``evidence_quote`` is present, long enough (>= 25 characters
                  and >= 4 words), and occurs verbatim (modulo whitespace, case
                  and typographic quotes) in the text the classifier was
                  actually given (``classifier_input``) — not the whole file

Diagnostics recorded on every check (never used to pass or reject):
  * ``offset``               where the quote sits in the window (for highlighting
                             and as an anchor for later stages)
  * ``longest_match_ratio``  longest contiguous overlap with the window / quote
                             length — separates "copied with a typo" from
                             "invented"
  * ``label_keyword_hit``    whether the quote contains vocabulary of its label
                             (a quote can exist and still not support the label)

Outcomes:
  - checks fail            → classify self-correction loop (error type is
                             written to the retry feedback); once the classify
                             budget is spent → DLQ with stage ``classify_validate``
  - checks pass, supported → continue to the Extractor
  - label ``none``         → end, DLQ code ``UNSUPPORTED_DOC_TYPE`` (out of scope)
  - label ``unclear``      → end, DLQ code ``CLASSIFICATION_UNCLEAR`` (needs a human)

``none`` / ``unclear`` are accepted as classifications (nothing to verify) but
they are terminal: the document has no extraction path.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Optional

from src.config import settings
from src.schemas.classification import (
    ALL_LABELS,
    OUT_OF_SCOPE_LABEL,
    SUPPORTED_DOC_TYPES,
    UNCLEAR_LABEL,
)
from src.nodes.retry import LOOP_CLASSIFY, register_classification_failure
from src.nodes.step_runs import (
    OUTCOME_FATAL,
    OUTCOME_PASSED,
    current_attempt,
    make_step_run,
    outcome_of,
)
from src.schemas.failure import (
    CAT_AMBIGUOUS,
    CAT_OUT_OF_SCOPE,
    CAT_REFERENCE,
    CAT_STRUCTURE,
    CLASSIFICATION_INVALID_LABEL,
    CLASSIFICATION_MISSING,
    CLASSIFICATION_UNCLEAR,
    EVIDENCE_QUOTE_MISSING,
    EVIDENCE_QUOTE_NOT_FOUND,
    EVIDENCE_QUOTE_TOO_SHORT,
    STAGE_CLASSIFY,
    STAGE_CLASSIFY_VALIDATE,
    UNSUPPORTED_DOC_TYPE,
    make_failure,
)
from src.schemas.state import PipelineState

# A quote shorter than this is too easy to match by accident ("the Merger").
MIN_QUOTE_CHARS = 25
MIN_QUOTE_WORDS = 4

_LABEL_KEYWORDS = {
    "M&A": re.compile(
        r"merg|acqui|combination|tender offer|divest|dispos|spin-?off|purchase",
        re.IGNORECASE,
    ),
    "Dividend": re.compile(r"dividend|distribution", re.IGNORECASE),
}

_TYPOGRAPHY = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def normalise_text(text: str) -> str:
    """Lower-case, unify typographic quotes/dashes and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _clean_quote(quote: str) -> str:
    """Normalise a quote and drop leading/trailing ellipses or quote marks."""
    quote = normalise_text(quote)
    return quote.strip(" .…\"'")


def quote_in_source(quote: str, source: str) -> bool:
    """True if ``quote`` occurs in ``source`` (after normalisation)."""
    cleaned = _clean_quote(quote)
    return bool(cleaned) and cleaned in normalise_text(source)


def quote_diagnostics(quote: str, source: str, label: Optional[str]) -> dict[str, Any]:
    """Where / how well the quote matches the window (diagnostics only)."""
    cleaned = _clean_quote(quote or "")
    window = normalise_text(source)
    offset = window.find(cleaned) if cleaned else -1
    ratio = 0.0
    if cleaned:
        if offset >= 0:
            ratio = 1.0
        else:
            m = SequenceMatcher(
                None, window, cleaned, autojunk=False
            ).find_longest_match(0, len(window), 0, len(cleaned))
            ratio = round(m.size / len(cleaned), 3)
    pattern = _LABEL_KEYWORDS.get(label or "")
    return {
        "found": offset >= 0,
        "offset": offset if offset >= 0 else None,
        "longest_match_ratio": ratio,
        "label_keyword_hit": bool(pattern.search(cleaned)) if pattern else None,
    }


# ── Individual checks: each returns (code, category, message) or None ──


def _check_label(label: Optional[str]) -> Optional[tuple[str, str, str]]:
    if label not in ALL_LABELS:
        return (
            CLASSIFICATION_INVALID_LABEL,
            CAT_STRUCTURE,
            f"[STRUCTURE] doc_type={label!r} is not one of {list(ALL_LABELS)}.",
        )
    return None


def _check_quote(quote: str, source: str) -> Optional[tuple[str, str, str]]:
    cleaned = _clean_quote(quote or "")
    if not cleaned:
        return (
            EVIDENCE_QUOTE_MISSING,
            CAT_STRUCTURE,
            "[STRUCTURE] evidence_quote is empty but the label requires a "
            "verbatim quote from the document.",
        )
    if len(cleaned) < MIN_QUOTE_CHARS or len(cleaned.split()) < MIN_QUOTE_WORDS:
        return (
            EVIDENCE_QUOTE_TOO_SHORT,
            CAT_STRUCTURE,
            f"[STRUCTURE] evidence_quote is shorter than {MIN_QUOTE_CHARS} "
            f"characters / {MIN_QUOTE_WORDS} words; quote the whole sentence.",
        )
    if cleaned not in normalise_text(source):
        shown = quote if len(quote) <= 80 else quote[:77] + "..."
        return (
            EVIDENCE_QUOTE_NOT_FOUND,
            CAT_REFERENCE,
            f"[REFERENCE] evidence_quote {shown!r} does not occur verbatim in "
            "the text you were given. Copy the exact words, do not paraphrase.",
        )
    return None


# ── Node function ──


async def classification_validator_node(state: PipelineState) -> dict:
    """
    LangGraph Node: validate the latest classification (see module docstring).

    Returns:
        State update with ``classification_passed`` and, on failure, the
        retry bookkeeping / terminal ``error`` + ``failure``.
    """
    # ── Short-circuit: skip if a previous node already errored ──
    if state.get("error"):
        return {}

    classification = state.get("classification")
    if not classification:
        return _reject(
            state,
            [
                (
                    CLASSIFICATION_MISSING,
                    CAT_STRUCTURE,
                    "[STRUCTURE] no classification found in state.",
                )
            ],
        )

    label = classification.get("label")
    quote = classification.get("evidence_quote", "")
    # The quote must come from what the model actually saw.
    window = state.get("classifier_input") or state.get("raw_content", "")
    diagnostics = quote_diagnostics(quote, window, label) if quote else None
    problems: list[tuple[str, str, str]] = []

    label_problem = _check_label(label)
    if label_problem:
        problems.append(label_problem)
    elif label in SUPPORTED_DOC_TYPES:
        quote_problem = _check_quote(quote, window)
        if quote_problem:
            problems.append(quote_problem)

    if problems:
        return _reject(state, problems, diagnostics)

    return _accept(state, label, diagnostics)


def _accept(
    state: PipelineState, label: str, diagnostics: Optional[dict] = None
) -> dict:
    """Classification is acceptable; terminal labels end the document here."""
    attempt = current_attempt(state, LOOP_CLASSIFY)
    if label in (OUT_OF_SCOPE_LABEL, UNCLEAR_LABEL):
        if label == OUT_OF_SCOPE_LABEL:
            message = (
                "classification: document is not a supported type "
                f"(label '{OUT_OF_SCOPE_LABEL}')."
            )
            failure = make_failure(
                STAGE_CLASSIFY, UNSUPPORTED_DOC_TYPE, CAT_OUT_OF_SCOPE, message
            )
        else:
            message = (
                "classification: the classifier could not decide "
                f"(label '{UNCLEAR_LABEL}') — needs human review."
            )
            failure = make_failure(
                STAGE_CLASSIFY, CLASSIFICATION_UNCLEAR, CAT_AMBIGUOUS, message
            )
        return {
            "classification_passed": True,
            "error": message,
            "failure": failure,
            "step_runs": [
                _step(
                    state,
                    attempt,
                    OUTCOME_FATAL,
                    output={"label": label, "checks": [], "quote": diagnostics},
                    failure=failure,
                )
            ],
        }
    return {
        "classification_passed": True,
        "step_runs": [
            _step(
                state,
                attempt,
                OUTCOME_PASSED,
                output={
                    "label": label,
                    "checks": ["label", "evidence_quote"],
                    "quote": diagnostics,
                },
            )
        ],
    }


def _reject(
    state: PipelineState,
    problems: list[tuple[str, str, str]],
    diagnostics: Optional[dict] = None,
) -> dict:
    """Spend one classify retry (or give up) and describe what was wrong."""
    attempt = current_attempt(state, LOOP_CLASSIFY)
    code, category, _ = problems[0]
    summary = "; ".join(msg for _, _, msg in problems)
    failures = [{"code": c, "category": cat, "message": m} for c, cat, m in problems]
    budget = state.get("classify_max_retries", settings.classify_max_retries)
    exhausted = (
        f"classification_validator: max retries ({budget}) exhausted. "
        f"Classification kept failing validation ({code})."
    )
    update = register_classification_failure(
        state,
        error_summary=summary,
        report="Classification validation FAILED:\n"
        + "\n".join(f"  - {msg}" for _, _, msg in problems),
        exhausted_message=exhausted,
        failure=make_failure(
            STAGE_CLASSIFY_VALIDATE,
            code,
            category,
            exhausted,
            details={"failures": failures},
        ),
    )
    update["classification_passed"] = False
    # The rejected label must not leak into the DLQ row as if it were accepted.
    update["doc_type"] = None
    update["step_runs"] = [
        _step(
            state,
            attempt,
            outcome_of(update),
            output={"failures": failures, "quote": diagnostics},
            error_code=code,
            error_category=category,
        )
    ]
    return update


def _step(state: PipelineState, attempt: int, outcome: str, **kw) -> dict:
    """Audit record for one classification check (no LLM involved)."""
    return make_step_run(
        state,
        stage=STAGE_CLASSIFY_VALIDATE,
        loop=LOOP_CLASSIFY,
        attempt=attempt,
        outcome=outcome,
        **kw,
    )
