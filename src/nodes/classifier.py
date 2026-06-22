"""
Classifier Node — uses LLM to categorise a raw financial announcement.

Outputs one of: "M&A", "Dividend", "Management_Change", or "Unknown".
This label determines which Extractor sub-graph is invoked downstream.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.state.pipeline_state import PipelineState


# ── Structured output schema for the LLM ──


class ClassificationResult(BaseModel):
    """Strict schema that the LLM must fill — no free-text allowed."""

    doc_type: str = Field(
        ...,
        description=(
            "One of: 'M&A', 'Dividend', 'Management_Change', or 'Unknown'. "
            "Choose the single best category for this announcement."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this classification (0.0 = unsure, 1.0 = certain).",
    )
    reasoning: str = Field(
        ...,
        max_length=300,
        description="Brief one-sentence justification for the classification.",
    )


# ── System prompt template ──

SYSTEM_PROMPT = """You are a financial document classifier for a US-market pipeline.
Your ONLY job is to categorise the announcement into exactly ONE of:

- M&A               — Merger, acquisition, asset purchase, tender offer, etc.
- Dividend          — Dividend declaration, distribution, payout ratio change.
- Management_Change — Executive appointment/resignation, board changes, C-suite moves.
- Unknown           — None of the above, or ambiguous.

Respond ONLY with valid JSON matching the ClassificationResult schema."""


# ── Node function ──


async def classifier_node(state: PipelineState) -> dict:
    """
    LangGraph Node: classify the raw announcement text.

    Reads ``raw_content`` from state, sends it to the LLM, and writes
    ``doc_type`` back into state.

    Args:
        state: Current PipelineState (must have ``raw_content`` populated).

    Returns:
        A dict of state updates: ``{"doc_type": ...}`` or ``{"error": ...}``.
    """
    # ── Short-circuit: skip if a previous node already errored ──
    if state.get("error"):
        return {}

    raw_content = state.get("raw_content", "").strip()
    if not raw_content:
        return {"error": "classifier_node: raw_content is empty"}

    client = LLMClient()
    try:
        result: ClassificationResult = await client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Classify the following announcement:\n\n{raw_content[:4000]}",
            response_model=ClassificationResult,
            temperature=0.05,  # near-deterministic for classification
            max_tokens=1024,
        )
    except Exception as exc:
        return {"error": f"classifier_node: LLM call failed — {exc}"}
    finally:
        await client.close()

    return {
        "doc_type": result.doc_type,
    }
