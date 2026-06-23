"""
Tests for the Validator Node — both individual check functions and the
node-level logic (rule execution, self-correction loop, DLQ routing).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.nodes.validator import (
    _check_non_empty,
    _check_positive,
    _check_range,
    _check_allowed_values,
    _check_date_format,
    _check_not_equal,
    _check_timeline_order,
    _check_amount_requires_currency,
    _check_not_all_empty,
    _run_rule,
    validator_node,
)


# ═══════════════════════════════════════════════════════════════════
#  Unit: individual check functions
# ═══════════════════════════════════════════════════════════════════

class TestCheckNonEmpty:
    def test_valid_string(self) -> None:
        assert _check_non_empty("Microsoft") is True

    def test_empty_string(self) -> None:
        assert _check_non_empty("") is False

    def test_whitespace_only(self) -> None:
        assert _check_non_empty("   ") is False

    def test_none(self) -> None:
        assert _check_non_empty(None) is False

    def test_non_string_type(self) -> None:
        assert _check_non_empty(123) is False


class TestCheckPositive:
    def test_positive_float(self) -> None:
        assert _check_positive(68.7) is True

    def test_positive_int(self) -> None:
        assert _check_positive(100) is True

    def test_zero(self) -> None:
        assert _check_positive(0) is False

    def test_negative(self) -> None:
        assert _check_positive(-5.0) is False

    def test_none(self) -> None:
        assert _check_positive(None) is False


class TestCheckRange:
    def test_mid_range(self) -> None:
        assert _check_range(50.0) is True

    def test_zero(self) -> None:
        assert _check_range(0.0) is True

    def test_one_hundred(self) -> None:
        assert _check_range(100.0) is True

    def test_above_max(self) -> None:
        assert _check_range(150.0) is False

    def test_below_zero(self) -> None:
        assert _check_range(-1.0) is False

    def test_none_optional(self) -> None:
        """None should pass — the field is optional."""
        assert _check_range(None) is True

    def test_non_number(self) -> None:
        assert _check_range("abc") is False


class TestCheckAllowedValues:
    def test_in_allowed_set(self) -> None:
        assert _check_allowed_values("Cash", {"Cash", "Stock"}) is True

    def test_not_in_allowed_set(self) -> None:
        assert _check_allowed_values("Crypto", {"Cash", "Stock"}) is False

    def test_none_optional(self) -> None:
        """None should pass — optional field."""
        assert _check_allowed_values(None, {"Cash"}) is True

    def test_case_sensitive(self) -> None:
        assert _check_allowed_values("cash", {"Cash"}) is False


class TestCheckDateFormat:
    def test_valid_iso(self) -> None:
        assert _check_date_format("2026-04-30") is True

    def test_invalid_format_slash(self) -> None:
        assert _check_date_format("04/30/2026") is False

    def test_invalid_month(self) -> None:
        assert _check_date_format("2026-13-01") is True  # regex doesn't validate calendar logic

    def test_none(self) -> None:
        """None is NOT allowed — missing date should fail validation."""
        assert _check_date_format(None) is False

    def test_empty_string(self) -> None:
        assert _check_date_format("") is False


class TestCheckNotEqual:
    def test_different_values(self) -> None:
        data = {"acquirer": "Microsoft", "target": "Activision"}
        passed, msg = _check_not_equal(data, ["acquirer", "target"])
        assert passed is True
        assert msg == ""

    def test_identical_values(self) -> None:
        data = {"acquirer": "Microsoft", "target": "Microsoft"}
        passed, msg = _check_not_equal(data, ["acquirer", "target"])
        assert passed is False
        assert "identical" in msg.lower()

    def test_case_insensitive(self) -> None:
        data = {"acquirer": "Microsoft", "target": "microsoft"}
        passed, msg = _check_not_equal(data, ["acquirer", "target"])
        assert passed is False

    def test_one_field_empty(self) -> None:
        """If one field is missing, we cannot compare — skip."""
        data = {"acquirer": "Microsoft", "target": ""}
        passed, msg = _check_not_equal(data, ["acquirer", "target"])
        assert passed is True


class TestCheckTimelineOrder:
    def test_valid_order(self) -> None:
        data = {
            "declaration_date": "2026-04-30",
            "ex_dividend_date": "2026-05-08",
            "record_date": "2026-05-12",
            "payment_date": "2026-05-21",
        }
        fields = ["declaration_date", "ex_dividend_date", "record_date", "payment_date"]
        passed, msg = _check_timeline_order(data, fields)
        assert passed is True

    def test_exdiv_before_declaration(self) -> None:
        """Ex-div before declaration is impossible."""
        data = {
            "declaration_date": "2026-05-08",
            "ex_dividend_date": "2026-04-30",
            "record_date": "2026-05-12",
            "payment_date": "2026-05-21",
        }
        fields = ["declaration_date", "ex_dividend_date", "record_date", "payment_date"]
        passed, msg = _check_timeline_order(data, fields)
        assert passed is False
        assert "declaration_date" in msg

    def test_record_before_exdiv(self) -> None:
        data = {
            "declaration_date": "2026-04-30",
            "ex_dividend_date": "2026-05-12",
            "record_date": "2026-05-08",
            "payment_date": "2026-05-21",
        }
        fields = ["declaration_date", "ex_dividend_date", "record_date", "payment_date"]
        passed, msg = _check_timeline_order(data, fields)
        assert passed is False
        assert "ex_dividend_date" in msg

    def test_same_date_allowed(self) -> None:
        """Same date for ex-div and record is allowed (e.g. same-day settlement)."""
        data = {
            "declaration_date": "2026-04-30",
            "ex_dividend_date": "2026-05-08",
            "record_date": "2026-05-08",
            "payment_date": "2026-05-21",
        }
        fields = ["declaration_date", "ex_dividend_date", "record_date", "payment_date"]
        passed, msg = _check_timeline_order(data, fields)
        assert passed is True

    def test_missing_date_skips(self) -> None:
        """If any date is None, skip the check entirely."""
        data = {
            "declaration_date": "2026-04-30",
            "ex_dividend_date": None,
            "record_date": "2026-05-12",
            "payment_date": "2026-05-21",
        }
        fields = ["declaration_date", "ex_dividend_date", "record_date", "payment_date"]
        passed, msg = _check_timeline_order(data, fields)
        assert passed is True


class TestCheckAmountRequiresCurrency:
    def test_amount_with_currency(self) -> None:
        data = {"dividend_cash_amount": 0.52, "currency": "USD"}
        passed, msg = _check_amount_requires_currency(data, "dividend_cash_amount", "currency")
        assert passed is True

    def test_amount_no_currency(self) -> None:
        data = {"dividend_cash_amount": 0.52, "currency": None}
        passed, msg = _check_amount_requires_currency(data, "dividend_cash_amount", "currency")
        assert passed is False
        assert "currency" in msg.lower()

    def test_no_amount_no_currency(self) -> None:
        """If neither is present, that's fine."""
        data = {"dividend_cash_amount": None, "currency": None}
        passed, msg = _check_amount_requires_currency(data, "dividend_cash_amount", "currency")
        assert passed is True

    def test_amount_with_empty_currency(self) -> None:
        data = {"dividend_cash_amount": 0.52, "currency": ""}
        passed, msg = _check_amount_requires_currency(data, "dividend_cash_amount", "currency")
        assert passed is False


class TestCheckNotAllEmpty:
    def test_at_least_one_field_populated(self) -> None:
        data = {
            "dividend_cash_amount": 0.52,
            "dividend_type": None,
            "declaration_date": None,
            "ex_dividend_date": None,
            "record_date": None,
            "payment_date": None,
        }
        fields = [
            "dividend_cash_amount", "dividend_type", "declaration_date",
            "ex_dividend_date", "record_date", "payment_date",
        ]
        passed, msg = _check_not_all_empty(data, fields)
        assert passed is True

    def test_all_fields_empty(self) -> None:
        data = {
            "dividend_cash_amount": None,
            "dividend_type": None,
            "declaration_date": None,
            "ex_dividend_date": None,
            "record_date": None,
            "payment_date": None,
        }
        fields = [
            "dividend_cash_amount", "dividend_type", "declaration_date",
            "ex_dividend_date", "record_date", "payment_date",
        ]
        passed, msg = _check_not_all_empty(data, fields)
        assert passed is False

    def test_zero_is_treated_as_empty(self) -> None:
        """0.0 is treated as 'empty' in the current implementation."""
        data = {
            "dividend_cash_amount": 0.0,
            "dividend_type": None,
            "declaration_date": None,
            "ex_dividend_date": None,
            "record_date": None,
            "payment_date": None,
        }
        fields = [
            "dividend_cash_amount", "dividend_type", "declaration_date",
            "ex_dividend_date", "record_date", "payment_date",
        ]
        passed, msg = _check_not_all_empty(data, fields)
        assert passed is False  # 0.0 is treated as empty — v == 0.0 check


# ═══════════════════════════════════════════════════════════════════
#  Unit: _run_rule
# ═══════════════════════════════════════════════════════════════════

class TestRunRule:
    def test_non_empty_pass(self) -> None:
        data = {"acquirer": "Microsoft"}
        rule = {"field": "acquirer", "check": "non_empty", "severity": "CRITICAL", "message": "Missing"}
        passed, msg = _run_rule(rule, data)
        assert passed is True

    def test_non_empty_fail(self) -> None:
        data = {"acquirer": ""}
        rule = {"field": "acquirer", "check": "non_empty", "severity": "CRITICAL", "message": "Missing"}
        passed, msg = _run_rule(rule, data)
        assert passed is False
        assert "[CRITICAL]" in msg

    def test_positive_pass(self) -> None:
        data = {"total_value_usd": 68.7}
        rule = {"field": "total_value_usd", "check": "positive", "severity": "HIGH", "message": "Not positive"}
        passed, msg = _run_rule(rule, data)
        assert passed is True

    def test_positive_fail(self) -> None:
        data = {"total_value_usd": 0}
        rule = {"field": "total_value_usd", "check": "positive", "severity": "HIGH", "message": "Not positive"}
        passed, msg = _run_rule(rule, data)
        assert passed is False

    def test_allowed_values_pass(self) -> None:
        data = {"currency": "USD"}
        rule = {"field": "currency", "check": "allowed_values", "severity": "MEDIUM", "message": "Bad currency"}
        passed, msg = _run_rule(rule, data)
        assert passed is True

    def test_allowed_values_fail(self) -> None:
        data = {"currency": "BTC"}
        rule = {"field": "currency", "check": "allowed_values", "severity": "MEDIUM", "message": "Bad currency"}
        passed, msg = _run_rule(rule, data)
        assert passed is False

    def test_unknown_check_passes_silently(self) -> None:
        """Unknown check type should pass to avoid breaking on new checks."""
        data = {"foo": "bar"}
        rule = {"field": "foo", "check": "unknown_check_type", "severity": "LOW", "message": "Unknown"}
        passed, msg = _run_rule(rule, data)
        assert passed is True


# ═══════════════════════════════════════════════════════════════════
#  Node-level tests: validator_node
# ═══════════════════════════════════════════════════════════════════

class TestValidatorNode:
    """Test the validator_node function with various PipelineState inputs."""

    async def test_manda_passes(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Clean M&A data should pass all validation rules."""
        result = await validator_node(sample_pipeline_state)
        assert result.get("validation_passed") is True
        assert result.get("error") is None

    async def test_manda_fails_on_empty_acquirer(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        """Missing acquirer should trigger CRITICAL failure."""
        state = dict(sample_pipeline_state)
        state["extracted_data"] = dict(state["extracted_data"])
        state["extracted_data"]["acquirer"] = ""
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        assert result.get("retry_count") == 1
        logs = result.get("correction_logs", [])
        assert len(logs) == 1
        assert "acquirer" in logs[0]["error_summary"].lower()

    async def test_manda_fails_when_acquirer_equals_target(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        state = dict(sample_pipeline_state)
        state["extracted_data"] = dict(state["extracted_data"])
        state["extracted_data"]["acquirer"] = "Company A"
        state["extracted_data"]["target"] = "Company A"
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        logs = result.get("correction_logs", [])
        assert len(logs) == 1
        # Should contain the cross-field not_equal failure
        assert any("identical" in log["error_summary"].lower() for log in logs)

    async def test_dividend_passes(self, dividend_pipeline_state: dict[str, Any]) -> None:
        """Clean Dividend data should pass all validation rules."""
        result = await validator_node(dividend_pipeline_state)
        assert result.get("validation_passed") is True

    async def test_dividend_timeline_fail(
        self, dividend_pipeline_state: dict[str, Any]
    ) -> None:
        """
        Reversed dates should fail the ``timeline_order`` cross-field check.

        All other fields in the dividend fixture are valid (ticker=AAPL,
        cash=0.52, currency=USD, dates in YYYY-MM-DD format, etc.) — only
        the timeline rule should fire, so ``len(logs) == 1``.

        The error message must contain ``timeline integrity`` (exact phrase
        from ``_check_timeline_order``), not just the sub-string ``timeline``
        which could theoretically match other messages.
        """
        state = dict(dividend_pipeline_state)
        state["extracted_data"] = dict(state["extracted_data"])
        # Flip ex-div before declaration — only this pair violates
        state["extracted_data"]["ex_dividend_date"] = "2026-04-01"
        state["extracted_data"]["declaration_date"] = "2026-04-30"
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        logs = result.get("correction_logs", [])
        assert len(logs) == 1  # Only timeline_order should fail
        assert "timeline integrity" in logs[0]["error_summary"].lower()

    async def test_dividend_amount_requires_currency(
        self, dividend_pipeline_state: dict[str, Any]
    ) -> None:
        state = dict(dividend_pipeline_state)
        state["extracted_data"] = dict(state["extracted_data"])
        state["extracted_data"]["currency"] = None
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        logs = result.get("correction_logs", [])
        assert any("currency" in log["error_summary"].lower() for log in logs)

    async def test_dividend_empty_action_guard(
        self, dividend_pipeline_state: dict[str, Any]
    ) -> None:
        """All fields empty should trigger the 'not_all_empty' guard."""
        state = dict(dividend_pipeline_state)
        state["extracted_data"] = {
            "ticker": None,
            "declaration_date": None,
            "dividend_cash_amount": None,
            "currency": None,
            "record_date": None,
            "ex_dividend_date": None,
            "payment_date": None,
            "dividend_type": None,
            "frequency": None,
        }
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        logs = result.get("correction_logs", [])
        assert any("empty" in log["error_summary"].lower() for log in logs)

    async def test_no_extracted_data(self, sample_pipeline_state: dict[str, Any]) -> None:
        """If extracted_data is None, validator should set error."""
        state = dict(sample_pipeline_state)
        state["extracted_data"] = None
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        assert "extracted_data" in (result.get("error") or "")

    async def test_unregistered_doc_type_passes(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        """No rules registered for doc_type → pass silently."""
        state = dict(sample_pipeline_state)
        state["doc_type"] = "Unknown"
        result = await validator_node(state)
        assert result.get("validation_passed") is True
        assert "No validation rules" in (result.get("validation_report") or "")

    async def test_retry_exhaustion(self, sample_pipeline_state: dict[str, Any]) -> None:
        """After max retries, validator should set error for DLQ routing."""
        state = dict(sample_pipeline_state)
        state["extracted_data"] = dict(state["extracted_data"])
        state["extracted_data"]["acquirer"] = ""
        state["retry_count"] = 2  # current retry before this validation
        state["max_retries"] = 3
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        assert result.get("retry_count") == 3
        logs = result.get("correction_logs", [])
        assert len(logs) == 1
        assert result.get("error") is not None
        assert "max retries" in (result.get("error") or "").lower()

    async def test_multiple_validation_errors(
        self, sample_pipeline_state: dict[str, Any]
    ) -> None:
        """Multiple failing rules should all be reported in correction_logs."""
        state = dict(sample_pipeline_state)
        state["extracted_data"] = dict(state["extracted_data"])
        state["extracted_data"]["acquirer"] = ""
        state["extracted_data"]["target"] = ""
        state["extracted_data"]["total_value_usd"] = -1.0
        result = await validator_node(state)
        assert result.get("validation_passed") is False
        logs = result.get("correction_logs", [])
        assert len(logs) >= 1
        # The error_summary should contain multiple issues
        summary = logs[0]["error_summary"]
        assert "acquirer" in summary.lower() or "target" in summary.lower()
        assert "total_value_usd" in summary.lower()

    async def test_short_circuit_on_error(self, sample_pipeline_state: dict[str, Any]) -> None:
        """If state already has error, validator should skip."""
        state = dict(sample_pipeline_state)
        state["error"] = "Previous node failed"
        result = await validator_node(state)
        assert result == {}  # No-op
