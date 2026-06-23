"""
Tests for schema models — Validation Profiles, BaseModel auto-injection,
and the two-model protocol (Model A → Model B promotion).
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import pytest
from pydantic import ValidationError

from src.schemas.base import BaseDoc, Metadata
from src.schemas.extraction.base_model import (
    BaseExtractionModel,
    ProfileValidatorMixin,
    _make_allowed_validator,
)
from src.schemas.extraction.profiles import (
    BaseValidationProfile,
    DividendValidationProfile,
    MandaValidationProfile,
)
from src.schemas.extraction.manda import MandaExtraction, MandaExtractionResult
from src.schemas.extraction.dividend import DividendExtraction, DividendExtractionResult


# ═══════════════════════════════════════════════════════════════════
#  Profile definitions
# ═══════════════════════════════════════════════════════════════════


class TestValidationProfiles:
    def test_dividend_profile_has_required_sets(self) -> None:
        assert "USD" in DividendValidationProfile.ALLOWED_CURRENCIES
        assert "Regular Cash" in DividendValidationProfile.ALLOWED_DIVIDEND_TYPES
        assert "Quarterly" in DividendValidationProfile.ALLOWED_FREQUENCIES
        assert "Semi-Annual" in DividendValidationProfile.ALLOWED_FREQUENCIES

    def test_manda_profile_has_required_sets(self) -> None:
        assert "Cash" in MandaValidationProfile.ALLOWED_PAYMENT_METHODS
        assert "Cash + Stock" in MandaValidationProfile.ALLOWED_PAYMENT_METHODS
        assert len(MandaValidationProfile.ALLOWED_PAYMENT_METHODS) == 5

    def test_profile_inheritance(self) -> None:
        """Create a hypothetical HK profile that inherits and adds CNH."""

        class HKProfile(DividendValidationProfile):
            ALLOWED_CURRENCIES = DividendValidationProfile.ALLOWED_CURRENCIES | {"CNH"}

        assert "CNH" in HKProfile.ALLOWED_CURRENCIES
        assert "USD" in HKProfile.ALLOWED_CURRENCIES  # inherited


# ═══════════════════════════════════════════════════════════════════
#  ProfileValidatorMixin & BaseExtractionModel
# ═══════════════════════════════════════════════════════════════════


class TestProfileValidatorMixin:
    def test_allowed_values_injected_automatically(self) -> None:
        """DividendExtraction should auto-validate currency, dividend_type, frequency."""
        # Valid data should pass
        model = DividendExtraction(
            ticker="AAPL",
            declaration_date="2026-04-30",
            dividend_cash_amount=0.52,
            currency="USD",
            record_date="2026-05-12",
            ex_dividend_date="2026-05-08",
            payment_date="2026-05-21",
            dividend_type="Regular Cash",
            frequency="Quarterly",
        )
        assert model.currency == "USD"
        assert model.dividend_type == "Regular Cash"

    def test_invalid_currency_raises(self) -> None:
        with pytest.raises(ValidationError, match="not in allowed set"):
            DividendExtraction(
                ticker="AAPL",
                declaration_date="2026-04-30",
                currency="BTC",  # not in allowed set
                dividend_type="Regular Cash",
                frequency="Quarterly",
            )

    def test_invalid_dividend_type_raises(self) -> None:
        with pytest.raises(ValidationError, match="not in allowed set"):
            DividendExtraction(
                ticker="AAPL",
                currency="USD",
                dividend_type="Super Dividend",  # not in allowed set
                frequency="Quarterly",
            )

    def test_invalid_frequency_raises(self) -> None:
        with pytest.raises(ValidationError, match="not in allowed set"):
            DividendExtraction(
                ticker="AAPL",
                currency="USD",
                dividend_type="Regular Cash",
                frequency="Weekly",  # not in allowed set
            )

    def test_manda_payment_method_valid(self) -> None:
        model = MandaExtraction(
            acquirer="Microsoft",
            target="Activision",
            payment_method="Cash",
        )
        assert model.payment_method == "Cash"

    def test_manda_payment_method_invalid(self) -> None:
        with pytest.raises(ValidationError, match="not in allowed set"):
            MandaExtraction(
                acquirer="Microsoft",
                target="Activision",
                payment_method="Crypto",  # not in allowed set
            )

    def test_none_passes_allowed_check(self) -> None:
        """Optional fields set to None should not trigger validation error."""
        model = DividendExtraction(
            ticker="AAPL",
            currency="USD",
            dividend_type="Regular Cash",
            frequency="Quarterly",
        )
        assert model.declaration_date is None  # not provided, still valid

    def test_date_format_auto_validated(self) -> None:
        """Any *_date field gets YYYY-MM-DD format check."""
        model = DividendExtraction(
            ticker="AAPL",
            declaration_date="2026-04-30",
            currency="USD",
            dividend_type="Regular Cash",
            frequency="Quarterly",
        )
        assert model.declaration_date == "2026-04-30"

    def test_date_format_invalid_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be YYYY-MM-DD"):
            DividendExtraction(
                ticker="AAPL",
                declaration_date="04/30/2026",  # slashes not allowed
                currency="USD",
                dividend_type="Regular Cash",
                frequency="Quarterly",
            )

    @pytest.mark.parametrize(
        "bad_date",
        [
            "2026-4-30",  # missing leading zero
            "2026/04/30",  # wrong separator
            "30-04-2026",  # DD-MM-YYYY
            "not-a-date",  # random text
            "",  # empty string
        ],
    )
    def test_various_bad_date_formats(self, bad_date: str) -> None:
        with pytest.raises(ValidationError, match="must be YYYY-MM-DD"):
            DividendExtraction(
                ticker="AAPL",
                declaration_date=bad_date,
                currency="USD",
                dividend_type="Regular Cash",
                frequency="Quarterly",
            )

    def test_dividend_cash_amount_negative_raises(self) -> None:
        """dividend_cash_amount < 0 should be rejected by ge=0.0 constraint."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DividendExtraction(
                ticker="AAPL",
                declaration_date="2026-04-30",
                dividend_cash_amount=-0.50,  # negative — violates ge=0.0
                currency="USD",
                dividend_type="Regular Cash",
                frequency="Quarterly",
            )


# ═══════════════════════════════════════════════════════════════════
#  Two-Model Protocol: Dividend
# ═══════════════════════════════════════════════════════════════════


class TestDividendTwoModelProtocol:
    def test_model_a_has_no_traceability_fields(self) -> None:
        """Model A (DividendExtraction) must NOT have doc_id or metadata."""
        fields = DividendExtraction.model_fields
        assert "doc_id" not in fields
        assert "metadata" not in fields

    def test_model_b_has_traceability_fields(self) -> None:
        """Model B (DividendExtractionResult) must have doc_id and metadata."""
        fields = DividendExtractionResult.model_fields
        assert "doc_id" in fields
        assert "metadata" in fields

    def test_promotion_from_extraction(self) -> None:
        """from_extraction() should correctly bridge Model A → Model B."""
        extraction = DividendExtraction(
            ticker="AAPL",
            declaration_date="2026-04-30",
            dividend_cash_amount=0.52,
            currency="USD",
            record_date="2026-05-12",
            ex_dividend_date="2026-05-08",
            payment_date="2026-05-21",
            dividend_type="Regular Cash",
            frequency="Quarterly",
        )
        metadata = Metadata(
            source_path="/test/apple.txt",
            model_name="deepseek-chat",
            doc_type="Dividend",
        )
        result = DividendExtractionResult.from_extraction(
            extraction=extraction,
            metadata=metadata,
            raw_text_snippet="Apple Inc. declares dividend...",
            doc_id="test-doc-id-123",
        )
        assert result.doc_id == "test-doc-id-123"
        assert result.metadata.source_path == "/test/apple.txt"
        assert result.ticker == "AAPL"
        assert result.dividend_cash_amount == 0.52
        assert result.raw_text_snippet == "Apple Inc. declares dividend..."

    def test_promotion_generates_doc_id(self) -> None:
        """If no doc_id provided, from_extraction should generate one."""
        extraction = DividendExtraction(
            ticker="AAPL",
            currency="USD",
            dividend_type="Regular Cash",
            frequency="Quarterly",
        )
        metadata = Metadata(
            source_path="/test/apple.txt",
            model_name="deepseek-chat",
            doc_type="Dividend",
        )
        result = DividendExtractionResult.from_extraction(
            extraction=extraction,
            metadata=metadata,
        )
        assert result.doc_id is not None
        assert len(result.doc_id) > 0  # Should be a UUID hex string


# ═══════════════════════════════════════════════════════════════════
#  Two-Model Protocol: M&A
# ═══════════════════════════════════════════════════════════════════


class TestMandaTwoModelProtocol:
    def test_model_a_has_no_traceability_fields(self) -> None:
        fields = MandaExtraction.model_fields
        assert "doc_id" not in fields
        assert "metadata" not in fields

    def test_model_b_has_traceability_fields(self) -> None:
        fields = MandaExtractionResult.model_fields
        assert "doc_id" in fields
        assert "metadata" in fields

    def test_promotion_from_extraction(self) -> None:
        extraction = MandaExtraction(
            acquirer="Microsoft Corporation",
            target="Activision Blizzard, Inc.",
            total_value_usd=68700000000.0,
            stake_percentage=100.0,
            requires_shareholder_approval=True,
            payment_method="Cash",
            announcement_date="2022-01-18",
            expected_close_date="2023-06-30",
        )
        metadata = Metadata(
            source_path="/test/msft_activision.txt",
            model_name="deepseek-chat",
            doc_type="M&A",
        )
        result = MandaExtractionResult.from_extraction(
            extraction=extraction,
            metadata=metadata,
            raw_text_snippet="Microsoft to acquire Activision...",
            doc_id="test-ma-doc-id",
        )
        assert result.doc_id == "test-ma-doc-id"
        assert result.acquirer == "Microsoft Corporation"
        assert result.target == "Activision Blizzard, Inc."
        assert result.total_value_usd == 68700000000.0
        assert result.payment_method == "Cash"
        assert result.announcement_date == "2022-01-18"


# ═══════════════════════════════════════════════════════════════════
#  BaseDoc & Metadata
# ═══════════════════════════════════════════════════════════════════


class TestBaseDoc:
    def test_metadata_requires_source_path(self) -> None:
        """Metadata.source_path is required."""
        with pytest.raises(ValidationError):
            Metadata(
                # missing source_path
                model_name="deepseek-chat",
                doc_type="M&A",
            )

    def test_metadata_retry_default(self) -> None:
        m = Metadata(
            source_path="/test/file.txt",
            model_name="deepseek-chat",
            doc_type="M&A",
        )
        assert m.retry_count == 0
        assert m.pipeline_version == "1.0.0"

    def test_doc_id_uuid_generated(self) -> None:
        """BaseDoc should auto-generate doc_id as UUID hex."""
        m = Metadata(
            source_path="/test/file.txt",
            model_name="deepseek-chat",
            doc_type="M&A",
        )
        doc = MandaExtractionResult(
            acquirer="A",
            target="B",
            metadata=m,
        )
        assert len(doc.doc_id) == 32  # UUID4 hex = 32 chars
        assert doc.doc_id.isalnum()


# ═══════════════════════════════════════════════════════════════════
#  _make_allowed_validator factory (edge cases)
# ═══════════════════════════════════════════════════════════════════


class TestMakeAllowedValidator:
    def test_injected_validators_from_mapped_profile(self) -> None:
        """ALLOWED_CURRENCIES maps to 'currency' field and auto-validates."""
        # DividendValidationProfile has ALLOWED_CURRENCIES → validates 'currency'
        # and ALLOWED_FREQUENCIES → validates 'frequency'
        m1 = DividendExtraction(
            ticker="AAPL",
            currency="USD",
            dividend_type="Regular Cash",
            frequency="Quarterly",
        )
        assert m1.currency == "USD"
        assert m1.frequency == "Quarterly"

    def test_sub_model_custom_profile(self) -> None:
        """
        A subclass with a different profile should use its own validators.

        ``ignored_types=(type,)`` is a standard Pydantic v2 config option
        (see https://docs.pydantic.dev/latest/api/config/) — it prevents
        the ``ValidationProfile`` class attribute (a ``type`` object) from
        being mistakenly treated as a model field.
        """

        class CustomProfile(BaseValidationProfile):
            ALLOWED_FRUITS = {"Apple", "Banana"}

        class CustomModel(BaseExtractionModel):
            model_config = {"ignored_types": (type,)}

            ValidationProfile = CustomProfile
            fruits: Optional[str] = None

        m = CustomModel(fruits="Apple")
        assert m.fruits == "Apple"

        with pytest.raises(ValidationError, match="not in allowed set"):
            CustomModel(fruits="Durian")
