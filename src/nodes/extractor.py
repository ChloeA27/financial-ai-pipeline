"""
Extractor Node — uses LLM to extract structured financial data.

For M&A documents, the LLM fills a lightweight ``MandaExtraction`` model
(pure business fields, all optional). The pipeline then promotes it into
``MandaExtractionResult`` with full traceability metadata.

This two-model pattern means:
  - The LLM never has to guess pipeline-controlled fields (doc_id, metadata, …)
  - If the LLM omits business fields, Pydantic defaults to None gracefully
  - The Validator later enforces business rules and triggers self-correction

---

**Registry Pattern** — adding a new document type is purely additive:

  1. Create schema in ``src/schemas/extraction/<new_type>.py``
  2. Import & register in the three dictionaries below
  3. Done — zero changes to the extraction logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.config import settings
from src.llm.client import LLMClient
from src.rag.retriever import retrieve_context
from src.schemas.base import Metadata
from src.schemas.extraction.dividend import DividendExtraction, DividendExtractionResult
from src.schemas.extraction.manda import MandaExtraction, MandaExtractionResult
from src.state.pipeline_state import PipelineState


# ════════════════════════════════════════════════════════════════════════
#  Registry — add a new doc_type by inserting ONE line in each dict
# ════════════════════════════════════════════════════════════════════════

EXTRACTOR_PROMPTS: dict[str, str] = {
    "M&A": """You are a financial data extractor for US SEC filings.
Extract the Merger & Acquisition details from the announcement below.

Respond with valid JSON containing ONLY these fields (all optional, use null if unknown):
- acquirer  = the buying entity (full legal name).
- target    = the company or assets being acquired.
- total_value_usd = numeric only (no $ signs or commas). Use null if not found.
- stake_percentage = numeric % (0.0–100.0). Use 100.0 for full acquisition.
- requires_shareholder_approval = true / false. Use null if not mentioned.
- payment_method = one of: "Cash", "Stock", "Cash + Stock", "Asset Swap", "Other".
- announcement_date = YYYY-MM-DD format.
- expected_close_date = YYYY-MM-DD format, or null if not stated.""",
    "Dividend": """You are a financial data extractor for corporate dividend announcements.
Extract the dividend distribution details from the announcement below.

Respond with valid JSON containing ONLY these fields (all optional, use null if unknown):
- ticker = stock ticker symbol (e.g. AAPL, MSFT).
- declaration_date = date the board declared the dividend in YYYY-MM-DD format.
- dividend_cash_amount = Gross (pre-tax) cash amount per share (numeric only, no currency symbols). Extract ONLY the pre-tax gross amount.
- currency = ISO 4217 three-letter currency code (e.g. USD, EUR, HKD).
- record_date = record / holders-of-record date in YYYY-MM-DD format.
- ex_dividend_date = ex-dividend date in YYYY-MM-DD format.
- payment_date = dividend payment / payable date in YYYY-MM-DD format.
- dividend_type = one of: "Regular Cash", "Special Cash", "Stock", "Property".
- frequency = one of: "Quarterly", "Monthly", "Semi-Annual", "Annual", "One-time".""",
}


def _get_default_prompt(doc_type: str) -> str:
    """Fallback prompt for unhandled document types."""
    return (
        f"Extract all structured financial data from this {doc_type} announcement. "
        "Respond with valid JSON containing only business fields."
    )


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

    raw_content = state.get("raw_content", "").strip()
    doc_type = state.get("doc_type", "")

    if not raw_content:
        return {"error": "extractor_node: raw_content is empty"}
    if not doc_type:
        return {"error": "extractor_node: doc_type is empty — run classifier first"}

    # Resolve the extraction model for this doc type
    try:
        llm_model = _resolve_model(doc_type)
        result_model = _resolve_result_model(doc_type)
    except ValueError as exc:
        return {"error": str(exc)}

    # Pick the system prompt
    system_prompt = EXTRACTOR_PROMPTS.get(doc_type, _get_default_prompt(doc_type))

    # ── Build the user prompt — include RAG context + correction feedback ──
    # RAG context: only on first pass (correction_feedback will handle retries)
    correction_feedback = _build_correction_feedback(state)
    user_prompt = f"Extract from this announcement:\n\n{raw_content[:6000]}"

    if not correction_feedback:
        # First pass → inject RAG context from historical extractions
        rag_context = await retrieve_context(
            doc_type=doc_type,
            query_text=raw_content[:1000],
        )
        if rag_context:
            user_prompt = rag_context + "\n" + user_prompt
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
            processed_at=datetime.now(timezone.utc),
            model_name=settings.deepseek_model,
            doc_type=doc_type,
            pipeline_version="1.0.0",
            retry_count=state.get("retry_count", 0),
        )

        final_result = result_model.from_extraction(
            extraction=llm_result,
            metadata=metadata,
            raw_text_snippet=raw_content[:500],
            doc_id=real_doc_id,
        )

        # ── Step 3: Serialise to dict for state ──
        return {
            "extracted_data": final_result.model_dump(mode="json"),
        }

    except Exception as exc:
        return {"error": f"extractor_node: LLM call failed — {exc}"}
    finally:
        await client.close()


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
