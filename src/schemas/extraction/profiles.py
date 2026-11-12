"""
Validation Profiles — single source of truth for allowed-value constraints.

Each business line (Dividend, M&A, …) gets its own Profile class.
Sub-profiles can inherit and override for business-line-specific rules.

Usage:
    class DividendExtraction(BaseExtractionModel):
        ValidationProfile = DividendValidationProfile

    class DividendExtractionResult(BaseDoc):
        ValidationProfile = DividendValidationProfile
"""

from __future__ import annotations

from typing import ClassVar


class BaseValidationProfile:
    """Base class for all validation profiles.

    Subclasses declare ``ClassVar[set[str]]`` fields following the
    naming convention ``ALLOWED_<PYDANTIC_FIELD_UPPER>``.

    Example:
        ``ALLOWED_CURRENCIES`` → validates Pydantic field ``currency``
        ``ALLOWED_FREQUENCIES`` → validates Pydantic field ``frequency``
    """

    # Presence levels (agreed 2026-11-12; evidence: results/field_presence/).
    # REQUIRED  — the record means nothing without it: a missing value is a
    #             missing_info failure and goes to human review.
    # EXPECTED  — usually disclosed; a missing value is allowed but flagged.
    # Every other field is OPTIONAL: absent is normal, format is checked only
    # when a value is present.
    # Contract only for now: the Extraction Validator does not read these yet.
    REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset()
    EXPECTED_FIELDS: ClassVar[frozenset[str]] = frozenset()


class DividendValidationProfile(BaseValidationProfile):
    """Dividend allowed-value constraints — single maintenance point."""

    # ex_dividend_date is OPTIONAL: it appears in 1 of 20 real dividend filings.
    REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"dividend_cash_amount", "declaration_date", "record_date", "payment_date"}
    )
    EXPECTED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"ticker", "currency", "dividend_type"}
    )

    ALLOWED_CURRENCIES: ClassVar[set[str]] = {
        "USD",
        "EUR",
        "CNY",
        "HKD",
        "GBP",
        "JPY",
        "CAD",
        "AUD",
        "SGD",
    }
    ALLOWED_DIVIDEND_TYPES: ClassVar[set[str]] = {
        "Regular Cash",
        "Special Cash",
        "Stock",
        "Property",
    }
    ALLOWED_FREQUENCIES: ClassVar[set[str]] = {
        "Quarterly",
        "Monthly",
        "Semi-Annual",
        "Annual",
        "One-time",
    }


class MandaValidationProfile(BaseValidationProfile):
    """M&A allowed-value constraints."""

    # total_value_usd is OPTIONAL: many deals do not disclose a value.
    REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset({"acquirer", "target"})
    EXPECTED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"announcement_date", "payment_method"}
    )

    ALLOWED_PAYMENT_METHODS: ClassVar[set[str]] = {
        "Cash",
        "Stock",
        "Cash + Stock",
        "Asset Swap",
        "Other",
    }
