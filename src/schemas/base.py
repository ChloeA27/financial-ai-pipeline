"""
Base document schema — shared foundation for all financial extraction models.

Every extraction output inherits from BaseDoc, ensuring **full traceability**
(Traceability / Auditability) as required by the Senior AI Data Engineer role.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Metadata(BaseModel):
    """Traceability metadata attached to every extraction document."""

    source_path: str = Field(
        ...,
        description="Absolute path to the original raw text file on disk.",
    )
    """Fully qualified filesystem path of the source announcement."""

    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO-8601 timestamp when this document was processed.",
    )
    """UTC timestamp of extraction."""

    model_name: str = Field(
        ...,
        description="Name of the LLM model that performed the extraction.",
    )
    """e.g. 'deepseek-chat', 'gpt-4o', 'claude-sonnet-4'."""

    doc_type: str = Field(
        ...,
        description="Classification label for this document, e.g. 'earnings'.",
    )
    """High-level category assigned by the Classifier Agent."""

    pipeline_version: str = Field(
        default="1.0.0",
        description="Semantic version of the extraction pipeline.",
    )
    """Used for audit trail — which version of the pipeline produced this result."""

    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of self-correction retries attempted.",
    )
    """Zero on first pass; incremented by Validator Agent on re-extraction."""

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all for any additional audit fields.",
    )
    """Flexible bucket for future metadata without schema migration."""


class BaseDoc(BaseModel):
    """
    Abstract base for all financial extraction documents.

    Every concrete extraction model (MandaExtraction, DividendExtraction, …)
    MUST inherit from this class to guarantee traceability compliance.
    """

    doc_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Globally unique identifier for this extraction document.",
    )
    """UUID4 hex string; generated at extraction time."""

    metadata: Metadata = Field(
        ...,
        description="Traceability metadata for audit trail.",
    )
    """Embedded Metadata object carrying full provenance."""

    raw_text_snippet: str = Field(
        default="",
        max_length=500,
        description="Short prefix of the original text for manual review.",
    )
    """First ~500 chars of the source file; aids human-in-the-loop debugging."""

    # ── Pydantic v2 configuration ──
    model_config = ConfigDict(frozen=False)  # Allow mutation during self-correction
