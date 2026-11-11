"""
Classification contract: the labels and the structured output the LLM must fill.

LLM-facing payload (like ``schemas/extraction/*`` for the Extractor). Kept out
of the Classifier node so the node, the Classification Validator, the graph
and the evaluation code all depend on one definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Types that have an extraction path.  Must stay in sync with the Extractor
# registry (enforced by a unit test).
SUPPORTED_DOC_TYPES: tuple[str, ...] = ("M&A", "Dividend")
OUT_OF_SCOPE_LABEL = "none"
UNCLEAR_LABEL = "unclear"
ALL_LABELS: tuple[str, ...] = SUPPORTED_DOC_TYPES + (OUT_OF_SCOPE_LABEL, UNCLEAR_LABEL)

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
