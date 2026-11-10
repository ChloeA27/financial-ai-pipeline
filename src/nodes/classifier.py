"""
Classifier Node — uses an LLM to label a raw financial announcement.

Labels (exactly one):
  - "M&A"      announces / reports an actual merger or acquisition
  - "Dividend" announces / reports an actual dividend
  - "none"     clearly neither (earnings, executive changes, rumours, ...)
  - "unclear"  might be one of the above but the model cannot decide

``none`` and ``unclear`` are deliberately separate: "none" is an expected,
out-of-scope outcome, while "unclear" means a human should look at it.

The model must also return a verbatim ``evidence_quote`` for M&A / Dividend.
The Classification Validator (plain code) checks that the quote really occurs
in the source text, which is a measurable signal that does not depend on the
model's self-reported ``confidence``.

An output that cannot be parsed (bad label, malformed JSON) is not fatal: it
goes through the classify self-correction loop with its own retry budget.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.config import settings
from src.llm.client import LLMCallInfo, LLMClient
from src.nodes.step_runs import (
    OUTCOME_FATAL,
    OUTCOME_PASSED,
    current_attempt,
    make_step_run,
    outcome_of,
)
from src.nodes.retry import (
    LOOP_CLASSIFY,
    latest_log,
    register_classification_failure,
    summarise_validation_error,
)
from src.state.failure import (
    CAT_INPUT_ERROR,
    CAT_STRUCTURE,
    CAT_TRANSIENT,
    CLASSIFICATION_PARSE_FAILED,
    EMPTY_DOCUMENT,
    LLM_CALL_FAILED,
    STAGE_CLASSIFY,
    make_failure,
)
from src.state.pipeline_state import PipelineState

# Types that have an extraction path.  Must stay in sync with the Extractor
# registry (enforced by a unit test).
SUPPORTED_DOC_TYPES: tuple[str, ...] = ("M&A", "Dividend")
OUT_OF_SCOPE_LABEL = "none"
UNCLEAR_LABEL = "unclear"
ALL_LABELS: tuple[str, ...] = SUPPORTED_DOC_TYPES + (OUT_OF_SCOPE_LABEL, UNCLEAR_LABEL)

CLASSIFIER_INPUT_CHARS = 4000


# ── Structured output schema for the LLM ──

DocLabel = Literal["M&A", "Dividend", "none", "unclear"]


class ClassificationResult(BaseModel):
    """Strict schema that the LLM must fill — the label is an enum, not free text."""

    doc_type: DocLabel = Field(
        ...,
        description="One of: 'M&A', 'Dividend', 'none', 'unclear'.",
    )
    evidence_quote: str = Field(
        default="",
        max_length=400,
        description=(
            "Verbatim quote (10-300 characters) copied from the document that "
            "justifies the label. Required for 'M&A' and 'Dividend'; empty for "
            "'none' and 'unclear'."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence (0.0-1.0). Logged only, never used for routing.",
    )
    reasoning: str = Field(
        ...,
        max_length=300,
        description="Brief one-sentence justification for the classification.",
    )


# ── System prompt ──

SYSTEM_PROMPT = """You are a financial document classifier for a US-market pipeline.
Assign the announcement to exactly ONE label:

- "M&A"      — the document announces or reports an ACTUAL merger, acquisition, asset purchase or tender offer (a definitive agreement, or a completed or formally announced deal).
- "Dividend" — the document announces or reports an ACTUAL dividend declaration or distribution.
- "none"     — the document is clearly neither: e.g. earnings results, executive or board changes, share buybacks, litigation, or rumours / analyst speculation about a possible deal.
- "unclear"  — the document may be M&A or Dividend but you cannot decide: e.g. it mixes several events, is truncated, or contradicts itself.

Also return:
- evidence_quote: for "M&A" and "Dividend", ONE verbatim quote of 10-300 characters copied exactly from the document (same words, same order, no paraphrasing, no added ellipses) that shows the event. For "none" and "unclear" use "".
- confidence: a number between 0 and 1.
- reasoning: one short sentence.

Respond ONLY with valid JSON matching the schema."""


# ── Node function ──


async def classifier_node(state: PipelineState) -> dict:
    """
    LangGraph Node: classify the raw announcement text.

    Reads ``raw_content`` from state, asks the LLM for a label plus evidence
    quote, and writes ``doc_type`` and ``classification`` back into state.

    Failure handling:
      - LLM / transport failure          → fatal ``error`` + ``failure`` (transient)
      - output fails the schema          → classify self-correction loop
        (``classification_passed=False``; fatal only when the budget is spent)

    Returns:
        A dict of state updates.
    """
    # ── Short-circuit: skip if a previous node already errored ──
    if state.get("error"):
        return {}

    attempt = current_attempt(state, LOOP_CLASSIFY)
    raw_content = state.get("raw_content", "").strip()
    if not raw_content:
        message = "classifier_node: raw_content is empty"
        failure = make_failure(STAGE_CLASSIFY, EMPTY_DOCUMENT, CAT_INPUT_ERROR, message)
        return {
            "error": message,
            "failure": failure,
            "step_runs": [_step(state, attempt, OUTCOME_FATAL, failure=failure)],
        }

    user_prompt = (
        "Classify the following announcement:\n\n"
        f"{raw_content[:CLASSIFIER_INPUT_CHARS]}"
    )
    feedback = _build_classification_feedback(state)
    if feedback:
        user_prompt += (
            "\n\n─── PREVIOUS CLASSIFICATION FEEDBACK (correct these issues) ───\n"
            f"{feedback}"
        )

    client = LLMClient()
    try:
        result: ClassificationResult = await client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ClassificationResult,
            temperature=0.05,  # near-deterministic for classification
            max_tokens=1024,
        )
    except ValidationError as exc:
        # The LLM answered, but not in the required shape — retry with feedback.
        return _parse_failure_update(state, exc, attempt, client.last_call)
    except Exception as exc:
        message = f"classifier_node: LLM call failed — {exc}"
        failure = make_failure(STAGE_CLASSIFY, LLM_CALL_FAILED, CAT_TRANSIENT, message)
        return {
            "error": message,
            "failure": failure,
            "step_runs": [
                _step(
                    state,
                    attempt,
                    OUTCOME_FATAL,
                    failure=failure,
                    llm_call=client.last_call,
                )
            ],
        }
    finally:
        await client.close()

    classification = {
        "label": result.doc_type,
        "evidence_quote": result.evidence_quote,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
    }
    return {
        "doc_type": result.doc_type,
        "classification": classification,
        # Reset so a stale False from a rejected attempt cannot reroute us.
        "classification_passed": None,
        "step_runs": [
            _step(
                state,
                attempt,
                OUTCOME_PASSED,
                output=classification,
                llm_call=client.last_call,
            )
        ],
    }


def _step(state: PipelineState, attempt: int, outcome: str, **kw) -> dict:
    """Audit record for one classifier call."""
    return make_step_run(
        state,
        stage=STAGE_CLASSIFY,
        loop=LOOP_CLASSIFY,
        attempt=attempt,
        outcome=outcome,
        **kw,
    )


# ── Schema-parse failure → self-correction ──


def _parse_failure_update(
    state: PipelineState,
    exc: ValidationError,
    attempt: int,
    llm_call: Optional[LLMCallInfo],
) -> dict:
    """State update for a classifier output that failed schema parsing."""
    summary = summarise_validation_error(exc, tag="SCHEMA")
    budget = state.get("classify_max_retries", settings.classify_max_retries)
    exhausted = (
        f"classifier_node: max retries ({budget}) exhausted. "
        "LLM output never matched the classification schema."
    )
    update = register_classification_failure(
        state,
        error_summary=summary,
        report=f"Classifier output failed schema parsing:\n  - {summary}",
        exhausted_message=exhausted,
        failure=make_failure(
            STAGE_CLASSIFY, CLASSIFICATION_PARSE_FAILED, CAT_STRUCTURE, exhausted
        ),
    )
    update["doc_type"] = None
    update["classification"] = None
    update["classification_passed"] = False
    update["step_runs"] = [
        _step(
            state,
            attempt,
            outcome_of(update),
            output={"schema_errors": summary},
            llm_call=llm_call,
            error_code=CLASSIFICATION_PARSE_FAILED,
            error_category=CAT_STRUCTURE,
        )
    ]
    return update


# ── Correction feedback builder ──


def _build_classification_feedback(state: PipelineState) -> str:
    """Feedback from the latest rejected classification attempt ('' on first pass)."""
    log = latest_log(state, LOOP_CLASSIFY)
    if log is None:
        return ""
    return (
        f"Classification attempt #{log['cycle']} was rejected.\n"
        f"Issues to fix:\n{log['error_summary']}\n\n"
        "Re-classify the document. evidence_quote must be copied verbatim from "
        "the document text. If you cannot find a quote that shows an actual "
        "M&A or Dividend event, answer 'none' or 'unclear' with an empty quote."
    )
