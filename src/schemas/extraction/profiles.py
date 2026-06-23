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

    pass


class DividendValidationProfile(BaseValidationProfile):
    """Dividend allowed-value constraints — single maintenance point."""

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

    ALLOWED_PAYMENT_METHODS: ClassVar[set[str]] = {
        "Cash",
        "Stock",
        "Cash + Stock",
        "Asset Swap",
        "Other",
    }
