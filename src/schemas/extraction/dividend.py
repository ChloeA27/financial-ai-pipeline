"""
Dividend extraction schema — cash dividend / distribution announcements.

Architecture — two-model pattern (mirrors M&A):

  DividendExtraction           ← LLM fills this (plain BaseModel, business only)
        │
        ▼
  DividendExtractionResult     ← Pipeline wraps with Metadata & traceability
                                  (inherits BaseDoc)

This keeps the LLM model clean — it never has to guess pipeline fields.
"""

from __future__ import annotations

import uuid
from typing import ClassVar, Optional

from pydantic import Field

from src.schemas.base import BaseDoc, Metadata
from src.schemas.extraction.base_model import (
    BaseExtractionModel,
    ProfileValidatorMixin,
)
from src.schemas.extraction.profiles import DividendValidationProfile


# ════════════════════════════════════════════════════════════════════════
#  Model A — Pure business fields, filled by the LLM
#  Inherits BaseExtractionModel which auto-injects BOTH:
#   1) Allowed-value validators from ``DividendValidationProfile``
#   2) YYYY-MM-DD format check for any ``*_date`` field
#  via ``ProfileValidatorMixin.__init_subclass__`` — zero manual validators.
# ════════════════════════════════════════════════════════════════════════


class DividendExtraction(BaseExtractionModel):
    """
    Dividend business fields — extracted by the LLM.

    NO ``doc_id`` here — that is a pipeline-controlled field generated
    in ``extractor.py`` to guarantee uniqueness.

    All fields are optional so the pipeline degrades gracefully on noisy
    or incomplete source text. The Validator later enforces business rules.

    All validation is automatic:
    - ``currency``, ``dividend_type``, ``frequency`` via ``ValidationProfile``
    - ``declaration_date``, ``record_date``, ``ex_dividend_date``,
      ``payment_date`` via ``*_date`` naming convention
    - ``ticker`` via Pydantic native ``min_length``/``max_length``
    - ``dividend_cash_amount`` via Pydantic native ``ge=0.0``

    Zero manually-defined ``@field_validator`` in this class.
    """

    ValidationProfile: ClassVar[type[DividendValidationProfile]] = (
        DividendValidationProfile
    )

    ticker: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol of the issuer, e.g. AAPL, MSFT.",
    )
    declaration_date: Optional[str] = Field(
        default=None,
        description="Declaration / announcement date when the board declared the dividend, in YYYY-MM-DD format.",
    )
    dividend_cash_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Gross (pre-tax) cash dividend amount per share in the declared currency. "
            "Extract ONLY the pre-tax gross amount, NOT the net amount after withholding tax."
        ),
    )
    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="ISO 4217 three-letter currency code, e.g. USD, EUR, HKD.",
    )
    record_date: Optional[str] = Field(
        default=None,
        description="Record date (holders-of-record date) in YYYY-MM-DD format.",
    )
    ex_dividend_date: Optional[str] = Field(
        default=None,
        description="Ex-dividend date in YYYY-MM-DD format.",
    )
    payment_date: Optional[str] = Field(
        default=None,
        description="Dividend payment / payable date in YYYY-MM-DD format.",
    )
    dividend_type: Optional[str] = Field(
        default=None,
        description="Type of distribution: 'Regular Cash', 'Special Cash', 'Stock', 'Property'.",
    )
    frequency: Optional[str] = Field(
        default=None,
        description="Payment frequency: 'Quarterly', 'Monthly', 'Semi-Annual', 'Annual', 'One-time'.",
    )


# ════════════════════════════════════════════════════════════════════════
#  Model B — Full result with pipeline-controlled Metadata overlay
#  Uses the SAME ValidationProfile via ProfileValidatorMixin so that
#  allowed-value rules NEVER go out of sync with Model A.
#
#  Date-format validation is also automatic via ``*_date`` convention.
# ════════════════════════════════════════════════════════════════════════


class DividendExtractionResult(BaseDoc, ProfileValidatorMixin):
    """
    Complete Dividend extraction result with full traceability metadata.

    Allowed-value validators (currency, dividend_type, frequency) are
    auto-injected via ``ProfileValidatorMixin`` using the SAME
    ``DividendValidationProfile`` as Model A — single source of truth.

    Date-format validation (declaration_date, record_date, etc.) is
    also auto-injected via the ``*_date`` naming convention.
    """

    ValidationProfile: ClassVar[type[DividendValidationProfile]] = (
        DividendValidationProfile
    )

    ticker: Optional[str] = Field(default=None)
    declaration_date: Optional[str] = Field(default=None)
    dividend_cash_amount: Optional[float] = Field(default=None, ge=0.0)
    currency: Optional[str] = Field(default=None)
    record_date: Optional[str] = Field(default=None)
    ex_dividend_date: Optional[str] = Field(default=None)
    payment_date: Optional[str] = Field(default=None)
    dividend_type: Optional[str] = Field(default=None)
    frequency: Optional[str] = Field(default=None)

    # Override BaseDoc's required metadata to optional for serialisation safety
    metadata: Optional[Metadata] = Field(default=None)

    @classmethod
    def from_extraction(
        cls,
        extraction: DividendExtraction,
        metadata: Metadata,
        raw_text_snippet: str = "",
        *,
        doc_id: str | None = None,
    ) -> DividendExtractionResult:
        """Promote LLM extraction into a full traceable result."""
        return cls(
            doc_id=doc_id if doc_id else uuid.uuid4().hex,
            metadata=metadata,
            raw_text_snippet=raw_text_snippet,
            ticker=extraction.ticker,
            declaration_date=extraction.declaration_date,
            dividend_cash_amount=extraction.dividend_cash_amount,
            currency=extraction.currency,
            record_date=extraction.record_date,
            ex_dividend_date=extraction.ex_dividend_date,
            payment_date=extraction.payment_date,
            dividend_type=extraction.dividend_type,
            frequency=extraction.frequency,
        )
