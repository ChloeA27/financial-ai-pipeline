"""
Extractor Node — uses LLM to extract structured financial data.

For M&A documents, the LLM fills a lightweight ``MandaExtraction`` model
(pure business fields, all optional). The pipeline then promotes it into
``MandaExtractionResult`` with full traceability metadata.

This two-model pattern means:
  - The LLM never has to guess pipeline-controlled fields (doc_id, metadata, …)
  - If the LLM omits business fields, Pydantic defaults to None gracefully
  - The Extraction Validator later enforces business rules and triggers self-correction

---

**Registry Pattern** — adding a new document type is purely additive:

  1. Create schema in ``src/schemas/extraction/<new_type>.py``
  2. Import & register in the three dictionaries below
  3. Done — zero changes to the extraction logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from src.config import settings
from src.llm.client import LLMCallInfo, LLMClient
from src.prompts import load_prompt
from src.rag.retriever import retrieve_context
from src.schemas.base import Metadata
from src.schemas.extraction.dividend import DividendExtraction, DividendExtractionResult
from src.schemas.extraction.manda import MandaExtraction, MandaExtractionResult
from src.nodes.retry import LOOP_EXTRACT, register_failure, summarise_validation_error
from src.nodes.step_runs import (
    OUTCOME_FATAL,
    OUTCOME_PASSED,
    current_attempt,
    make_step_run,
    outcome_of,
)
from src.schemas.failure import (
    CAT_INPUT_ERROR,
    CAT_OUT_OF_SCOPE,
    CAT_STRUCTURE,
    CAT_TRANSIENT,
    EMPTY_DOCUMENT,
    LLM_CALL_FAILED,
    MISSING_DOC_TYPE,
    SCHEMA_PARSE_FAILED,
    STAGE_EXTRACT,
    UNSUPPORTED_DOC_TYPE,
    make_failure,
)
from src.schemas.state import PipelineState


# ════════════════════════════════════════════════════════════════════════
#  Registry — add a new doc_type by inserting ONE line in each dict
# ════════════════════════════════════════════════════════════════════════

# System prompts are assets: src/prompts/templates/extractor_<type>_system.md
EXTRACTOR_PROMPTS: dict[str, str] = {
    "M&A": load_prompt("extractor_ma_system"),
    "Dividend": load_prompt("extractor_dividend_system"),
}


def _get_default_prompt(doc_type: str) -> str:
    """Fallback prompt for unhandled document types."""
    return load_prompt("extractor_default_system", doc_type=doc_type)


# ── LLM-facing model (pure business fields, no traceability) ──

EXTRACTOR_MODELS: dict[str, type] = {
    "M&A": MandaExtraction,
    "Dividend": DividendExtraction,
}


# ── Pipeline-facing result model (with Metadata overlay) ──

EXTRACTOR_RESULT_MODELS: dict[str, type] = {
    "M&A": MandaExtractionResult,
    "Dividend": DividendExtractionResult,
}


def _resolve_model(doc_type: str) -> type:
    """Return the LLM-facing Pydantic model class for a given doc_type."""
    model = EXTRACTOR_MODELS.get(doc_type)
    if model is None:
        raise ValueError(
            f"extractor_node: no model registered for doc_type='{doc_type}'"
        )
    return model


def _resolve_result_model(doc_type: str) -> type:
    """Return the pipeline-facing result model class for a given doc_type."""
    model = EXTRACTOR_RESULT_MODELS.get(doc_type)
    if model is None:
        raise ValueError(
            f"extractor_node: no result model registered for doc_type='{doc_type}'"
        )
    return model


# ── Node function ──


async def extractor_node(state: PipelineState) -> dict:
    """
    LangGraph Node: extract structured data using the LLM.

    Reads ``raw_content`` + ``doc_type`` from state, invokes the
    appropriate extraction model, and writes the serialised result
    into ``extracted_data``.

    Args:
        state: PipelineState with ``raw_content`` and ``doc_type`` populated.

    Returns:
        Dict with ``extracted_data`` (serialised Pydantic model) or ``error``.
    """
    # ── Short-circuit: skip if a previous node already errored ──
    if state.get("error"):
        return {}

    attempt = current_attempt(state, LOOP_EXTRACT)
    raw_content = state.get("raw_content", "").strip()
    doc_type = state.get("doc_type", "")

    if not raw_content:
        return _fatal(
            state,
            attempt,
            EMPTY_DOCUMENT,
            CAT_INPUT_ERROR,
            "extractor_node: raw_content is empty",
        )
    if not doc_type:
        return _fatal(
            state,
            attempt,
            MISSING_DOC_TYPE,
            CAT_INPUT_ERROR,
            "extractor_node: doc_type is empty — run classifier first",
        )

    # Resolve the extraction model for this doc type
    try:
        llm_model = _resolve_model(doc_type)
        result_model = _resolve_result_model(doc_type)
    except ValueError as exc:
        return _fatal(state, attempt, UNSUPPORTED_DOC_TYPE, CAT_OUT_OF_SCOPE, str(exc))

    # Pick the system prompt
    system_prompt = EXTRACTOR_PROMPTS.get(doc_type, _get_default_prompt(doc_type))

    # ── Build the user prompt — include RAG context + correction feedback ──
    # RAG context: only on first pass (correction_feedback will handle retries)
    correction_feedback = _build_correction_feedback(state)
    user_prompt = f"Extract from this announcement:\n\n{raw_content[:6000]}"
    rag_context_used = False

    if not correction_feedback:
        # First pass → inject RAG context from historical extractions (if enabled)
        rag_context = ""
        if settings.rag_enabled:
            rag_context = await retrieve_context(
                doc_type=doc_type,
                query_text=raw_content[:1000],
                exclude_file_path=state.get("file_path") or None,
            )
        if rag_context:
            user_prompt = rag_context + "\n" + user_prompt
            rag_context_used = True
    else:
        user_prompt += (
            "\n\n─── PREVIOUS VALIDATION FEEDBACK (correct these issues) ───\n"
            f"{correction_feedback}"
        )

    client = LLMClient()
    try:
        # ── Step 1: LLM fills business fields only ──
        llm_result = await client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=llm_model,
            temperature=0.1,
            max_tokens=4096,
        )

        # ── Step 2: Pipeline generates a REAL UUID doc_id + overlays metadata ──
        real_doc_id = uuid.uuid4().hex

        metadata = Metadata(
            source_path=state.get("file_path", ""),
            source_hash=state.get("source_hash") or None,
            processed_at=datetime.now(timezone.utc),
            model_name=settings.deepseek_model,
            doc_type=doc_type,
            pipeline_version="1.0.0",
            retry_count=state.get("retry_count", 0),
        )

        final_result = result_model.from_extraction(
            extraction=llm_result,
            metadata=metadata,
            doc_id=real_doc_id,
        )

        # ── Step 3: Serialise to dict for state ──
        # validation_passed is reset so a stale False from a previous
        # attempt can't make the post-extraction router skip the Extraction Validator.
        return {
            "extracted_data": final_result.model_dump(mode="json"),
            "validation_passed": None,
            "step_runs": [
                _step(
                    state,
                    attempt,
                    OUTCOME_PASSED,
                    output={
                        "fields": llm_result.model_dump(mode="json"),
                        "rag_context_used": rag_context_used,
                    },
                    llm_call=client.last_call,
                )
            ],
        }

    except ValidationError as exc:
        # The LLM answered, but its output broke the schema (bad enum, date
        # format, out-of-range number, malformed JSON). That is exactly what
        # the self-correction loop is for, so don't abort — log and retry.
        return _parse_failure_update(state, exc, attempt, client.last_call)
    except Exception as exc:
        return _fatal(
            state,
            attempt,
            LLM_CALL_FAILED,
            CAT_TRANSIENT,
            f"extractor_node: LLM call failed — {exc}",
            llm_call=client.last_call,
        )
    finally:
        await client.close()


# ── Schema-parse failure → self-correction ──


def _fatal(
    state: PipelineState,
    attempt: int,
    code: str,
    category: str,
    message: str,
    llm_call: Optional[LLMCallInfo] = None,
) -> dict:
    """State update for an unrecoverable extractor failure (→ DLQ)."""
    failure = make_failure(STAGE_EXTRACT, code, category, message)
    return {
        "error": message,
        "failure": failure,
        "step_runs": [
            _step(state, attempt, OUTCOME_FATAL, failure=failure, llm_call=llm_call)
        ],
    }


def _step(state: PipelineState, attempt: int, outcome: str, **kw) -> dict:
    """Audit record for one extractor call."""
    return make_step_run(
        state,
        stage=STAGE_EXTRACT,
        loop=LOOP_EXTRACT,
        attempt=attempt,
        outcome=outcome,
        **kw,
    )


# Kept as a module-level alias: the helper now lives in ``retry`` so the
# Classifier can share it.
_summarise_validation_error = summarise_validation_error


def _parse_failure_update(
    state: PipelineState,
    exc: ValidationError,
    attempt: int,
    llm_call: Optional[LLMCallInfo],
) -> dict:
    """
    Build the state update for an LLM output that failed schema parsing.

    Goes through the same ``register_failure`` bookkeeping as the Extraction Validator,
    so the router treats both kinds of failure identically.
    """
    summary = summarise_validation_error(exc)
    max_retries = state.get("max_retries", 3)
    exhausted = (
        f"extractor_node: max retries ({max_retries}) exhausted. "
        "LLM output never matched the extraction schema."
    )
    update = register_failure(
        state,
        error_summary=summary,
        report=f"Extraction output failed schema parsing:\n  - {summary}",
        exhausted_message=exhausted,
        failure=make_failure(
            STAGE_EXTRACT, SCHEMA_PARSE_FAILED, CAT_STRUCTURE, exhausted
        ),
    )
    update["extracted_data"] = None
    update["validation_passed"] = False
    update["validation_report"] = (
        f"Extraction output failed schema parsing:\n  - {summary}"
    )
    update["step_runs"] = [
        _step(
            state,
            attempt,
            outcome_of(update),
            output={"schema_errors": summary},
            llm_call=llm_call,
            error_code=SCHEMA_PARSE_FAILED,
            error_category=CAT_STRUCTURE,
        )
    ]
    return update


# ── Correction feedback builder ──


def _build_correction_feedback(state: PipelineState) -> str:
    """
    Assemble validation feedback from previous correction cycles.

    If this is the first pass (no corrections yet), returns empty string.
    Otherwise, returns a concise summary of what went wrong in the last
    attempt so the LLM can fix it on re-extraction.
    """
    logs = state.get("correction_logs", [])
    if not logs:
        return ""

    latest = logs[-1]
    cycle = latest.get("cycle", 0)
    summary = latest.get("error_summary", "Unknown validation errors.")

    return (
        f"Correction cycle #{cycle} — previous extraction failed validation.\n"
        f"Issues to fix:\n{summary}\n\n"
        "Please re-extract the data, paying close attention to these fields. "
        "If the information is genuinely absent from the source text, "
        "set the field to null — do not fabricate data."
    )
