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
from src.nodes.input_window import build_input
from src.nodes.step_runs import (
    OUTCOME_FATAL,
    OUTCOME_PASSED,
    current_attempt,
    make_step_run,
    outcome_of,
)
from src.prompts import load_prompt
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
            "Verbatim sentence (25-300 characters) copied from the document that "
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


# ── System prompt (asset: src/prompts/templates/classifier_system.md) ──

SYSTEM_PROMPT = load_prompt("classifier_system")


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

    window = build_input(
        raw_content,
        strategy=settings.classifier_input_strategy,
        budget_tokens=settings.classifier_input_tokens,
    )
    temperature = settings.classify_temperature
    call_info = {
        **window.stats(),
        "temperature": temperature,
        "thinking": settings.classify_thinking or "provider default",
        "max_tokens": settings.classify_max_tokens,
    }
    user_prompt = f"Classify the following announcement:\n\n{window.text}"
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
            temperature=temperature,
            max_tokens=settings.classify_max_tokens,
            thinking=settings.classify_thinking,
        )
    except ValidationError as exc:
        # The LLM answered, but not in the required shape — retry with feedback.
        call_info["reasoning_tokens"] = (client.last_call or {}).get("reasoning_tokens")
        update = _parse_failure_update(state, exc, attempt, client.last_call)
        update["classifier_input"] = window.text
        update["step_runs"][0]["output"]["input"] = call_info
        return update
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
                    output={"input": call_info},
                    failure=failure,
                    llm_call=client.last_call,
                )
            ],
        }
    finally:
        await client.close()

    call_info["reasoning_tokens"] = (client.last_call or {}).get("reasoning_tokens")
    classification = {
        "label": result.doc_type,
        "evidence_quote": result.evidence_quote,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
    }
    return {
        "doc_type": result.doc_type,
        "classification": classification,
        "classifier_input": window.text,
        # Reset so a stale False from a rejected attempt cannot reroute us.
        "classification_passed": None,
        "step_runs": [
            _step(
                state,
                attempt,
                OUTCOME_PASSED,
                output={**classification, "input": call_info},
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
