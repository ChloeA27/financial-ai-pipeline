"""
M&A (Merger & Acquisition) extraction schema — US-market / SEC filings edition.

Architecture — two-model pattern:

  MandaExtraction           ← LLM fills this (plain BaseModel, business only)
        │
        ▼
  MandaExtractionResult     ← Pipeline wraps with Metadata & traceability
                               (inherits BaseDoc)

This keeps the LLM model clean — it never has to guess pipeline fields.
"""

from __future__ import annotations

import uuid
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

from src.schemas.base import BaseDoc, Metadata
from src.schemas.extraction.base_model import (
    BaseExtractionModel,
    ProfileValidatorMixin,
)
from src.schemas.extraction.profiles import MandaValidationProfile


# ════════════════════════════════════════════════════════════════════════
#  Model A — Pure business fields, filled by the LLM
#  Inherits BaseExtractionModel which auto-injects BOTH:
#   1) Allowed-value validator for ``payment_method`` from
#      ``MandaValidationProfile``
#   2) YYYY-MM-DD format check for ``announcement_date`` and
#      ``expected_close_date`` via ``*_date`` naming convention
# ════════════════════════════════════════════════════════════════════════


class MandaExtraction(BaseExtractionModel):
    """
    M&A business fields — extracted by the LLM.
    NO ``doc_id`` here — that is a pipeline-controlled field generated
    in ``extractor.py`` to guarantee uniqueness.

    All fields are optional so the pipeline degrades gracefully on noisy
    or incomplete source text. The Validator later enforces business rules.

    All validation is automatic:
    - ``payment_method`` via ``MandaValidationProfile``
    - ``announcement_date``, ``expected_close_date`` via ``*_date`` naming convention
    - ``total_value_usd``, ``stake_percentage`` via Pydantic native ``ge``/``le``

    Zero manually-defined ``@field_validator`` in this class.
    """

    ValidationProfile: ClassVar[type[MandaValidationProfile]] = MandaValidationProfile

    acquirer: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Name of the acquiring entity (the buyer).",
    )
    target: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Name of the target company being acquired.",
    )
    total_value_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total transaction enterprise value in US Dollars.",
    )
    stake_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percentage of equity being acquired (0.0–100.0).",
    )
    requires_shareholder_approval: Optional[bool] = Field(
        default=None,
        description="Whether the deal requires a shareholder vote.",
    )
    payment_method: Optional[str] = Field(
        default=None,
        description="Consideration: Cash, Stock, Cash + Stock, Asset Swap, Other.",
    )
    announcement_date: Optional[str] = Field(
        default=None,
        description="Public announcement date in YYYY-MM-DD format.",
    )
    expected_close_date: Optional[str] = Field(
        default=None,
        description="Expected closing date in YYYY-MM-DD format.",
    )


# ════════════════════════════════════════════════════════════════════════
#  Model B — Full result with pipeline-controlled Metadata overlay
#  Uses the SAME ValidationProfile via ProfileValidatorMixin so that
#  allowed-value rules NEVER go out of sync with Model A.
# ════════════════════════════════════════════════════════════════════════


class MandaExtractionResult(BaseDoc, ProfileValidatorMixin):
    """
    Complete M&A extraction result with full traceability metadata.

    Allowed-value validator for ``payment_method`` is auto-injected via
    ``ProfileValidatorMixin`` using the SAME ``MandaValidationProfile``
    as Model A — single source of truth.

    Date-format validation (``announcement_date``, ``expected_close_date``)
    is also auto-injected via the ``*_date`` naming convention.
    """

    ValidationProfile: ClassVar[type[MandaValidationProfile]] = MandaValidationProfile

    acquirer: Optional[str] = Field(default=None)
    target: Optional[str] = Field(default=None)
    total_value_usd: Optional[float] = Field(default=None, ge=0.0)
    stake_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    requires_shareholder_approval: Optional[bool] = Field(default=None)
    payment_method: Optional[str] = Field(default=None)
    announcement_date: Optional[str] = Field(default=None)
    expected_close_date: Optional[str] = Field(default=None)

    # Override BaseDoc's required metadata to optional for serialisation safety
    # for backwards compatibility with older pipeline code that doesn't generate Metadata
    metadata: Optional[Metadata] = Field(default=None)

    @classmethod
    def from_extraction(
        cls,
        extraction: MandaExtraction,
        metadata: Metadata,
        raw_text_snippet: str = "",
        *,
        doc_id: str | None = None,
    ) -> MandaExtractionResult:
        """Promote LLM extraction into a full traceable result."""
        return cls(
            doc_id=doc_id if doc_id else uuid.uuid4().hex,
            metadata=metadata,
            raw_text_snippet=raw_text_snippet,
            acquirer=extraction.acquirer,
            target=extraction.target,
            total_value_usd=extraction.total_value_usd,
            stake_percentage=extraction.stake_percentage,
            requires_shareholder_approval=extraction.requires_shareholder_approval,
            payment_method=extraction.payment_method,
            announcement_date=extraction.announcement_date,
            expected_close_date=extraction.expected_close_date,
        )
