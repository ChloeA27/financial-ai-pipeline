"""
Validator Node — doc_type-aware rule-based quality gate for extraction results.

Each document type has its own set of validation rules registered in
``_VALIDATION_RULESETS``. When no ruleset exists for a given doc_type,
the validator **passes silently** — because Pydantic schema validation
already caught structural errors at parse time.

Architecture — Registry Pattern (mirrors the Extractor):
  - Add a new ruleset dict for a new doc_type
  - The Validator Node function is generic — zero code changes

On failure: increments retry_count, appends to correction_logs,
and provides structured feedback for the next extraction attempt.
"""

from __future__ import annotations

from typing import Any

from src.state.pipeline_state import PipelineState, CorrectionLog


# ====================================================================
#  Validation rule sets — one list per document type
# ====================================================================

_VALIDATION_RULESETS: dict[str, list[dict[str, Any]]] = {
    # ── M&A rules ────────────────────────────────────────────────
    "M&A": [
        # Single-field checks
        {
            "field": "acquirer",
            "check": "non_empty",
            "severity": "CRITICAL",
            "message": "Acquirer (buyer) name is missing or empty.",
        },
        {
            "field": "target",
            "check": "non_empty",
            "severity": "CRITICAL",
            "message": "Target (seller) name is missing or empty.",
        },
        {
            "field": "total_value_usd",
            "check": "positive",
            "severity": "HIGH",
            "message": (
                "Total transaction value (total_value_usd) is "
                "missing, zero, or negative."
            ),
        },
        {
            "field": "stake_percentage",
            "check": "range",
            "severity": "MEDIUM",
            "message": "Stake percentage is outside the valid range (0.0–100.0).",
        },
        {
            "field": "payment_method",
            "check": "allowed_values",
            "severity": "LOW",
            # allowed set handled dynamically in _run_rule
            "message": (
                "Payment method is not in the allowed set: "
                "Cash, Stock, Cash + Stock, Asset Swap, Other."
            ),
        },
        # Cross-field checks
        {
            "field": "__cross__acquirer_vs_target",
            "check": "not_equal",
            "severity": "CRITICAL",
            "message": (
                "Acquirer (buyer) and Target (seller) are identical"
                " — likely an extraction error."
            ),
            "fields": ["acquirer", "target"],
        },
    ],
    # ── Dividend rules ───────────────────────────────────────────
    "Dividend": [
        {
            "field": "ticker",
            "check": "non_empty",
            "severity": "CRITICAL",
            "message": "Ticker symbol is missing or empty.",
        },
        {
            "field": "declaration_date",
            "check": "date_format",
            "severity": "HIGH",
            "message": "Declaration / announcement date is missing or not in YYYY-MM-DD format.",
        },
        {
            "field": "dividend_cash_amount",
            "check": "positive",
            "severity": "HIGH",
            "message": (
                "Dividend cash amount per share is missing, "
                "zero, or negative."
            ),
        },
        {
            "field": "currency",
            "check": "allowed_values",
            "severity": "MEDIUM",
            "message": (
                "Currency is not a supported ISO 4217 code: "
                "USD, EUR, CNY, HKD, GBP, JPY, CAD, AUD, SGD."
            ),
            # allowed set handled dynamically in _run_rule
        },
        {
            "field": "record_date",
            "check": "date_format",
            "severity": "MEDIUM",
            "message": "Record date is missing or not in YYYY-MM-DD format.",
        },
        {
            "field": "ex_dividend_date",
            "check": "date_format",
            "severity": "MEDIUM",
            "message": "Ex-dividend date is missing or not in YYYY-MM-DD format.",
        },
        {
            "field": "payment_date",
            "check": "date_format",
            "severity": "LOW",
            "message": "Payment date is missing or not in YYYY-MM-DD format.",
        },
        {
            "field": "dividend_type",
            "check": "allowed_values",
            "severity": "LOW",
            "message": (
                "Dividend type is not in the allowed set: "
                "Regular Cash, Special Cash, Stock, Property."
            ),
        },
        {
            "field": "frequency",
            "check": "allowed_values",
            "severity": "LOW",
            "message": (
                "Frequency is not in the allowed set: "
                "Quarterly, Monthly, Annual, Semi-Annual, One-time."
            ),
        },
    ],
}


# ====================================================================
#  Allowed-values lookup table (by field name)
# ====================================================================

_ALLOWED_VALUES: dict[str, set[str]] = {
    "payment_method": {"Cash", "Stock", "Cash + Stock", "Asset Swap", "Other"},
    "currency": {"USD", "EUR", "CNY", "HKD", "GBP", "JPY", "CAD", "AUD", "SGD"},
    "dividend_type": {"Regular Cash", "Special Cash", "Stock", "Property"},
    "frequency": {"Quarterly", "Monthly", "Annual", "Semi-Annual", "One-time"},
}


# ====================================================================
#  Check functions
# ====================================================================


def _check_non_empty(value: Any) -> bool:
    """Value must be a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def _check_positive(value: Any) -> bool:
    """Value must be a positive number (not None, not 0, not negative)."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value > 0.0
    return False


def _check_range(value: Any) -> bool:
    """Value must be None or in 0.0–100.0."""
    if value is None:
        return True  # Optional field — skip
    if isinstance(value, (int, float)):
        return 0.0 <= value <= 100.0
    return False


def _check_allowed_values(value: Any, allowed: set[str]) -> bool:
    """Value must be None or in the allowed set."""
    if value is None:
        return True  # Optional field — skip
    return value in allowed


def _check_date_format(value: Any) -> bool:
    """Value must be None or match YYYY-MM-DD."""
    if value is None:
        return False  # Missing date is a validation failure
    import re

    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)))


def _check_not_equal(data: dict[str, Any], fields: list[str]) -> tuple[bool, str]:
    """
    Cross-field check: two fields must NOT be equal (case-insensitive).

    Catches cases where the LLM accidentally copies the same name
    into both acquirer and target.
    """
    v1 = (data.get(fields[0]) or "").strip().lower()
    v2 = (data.get(fields[1]) or "").strip().lower()

    if not v1 or not v2:
        return True, ""  # Can't compare if one is missing

    if v1 == v2:
        return False, (
            f"[CRITICAL] Cross-field check failed: '{fields[0]}' == '{fields[1]}' "
            f"(both are '{data.get(fields[0])}'). "
            f"Values cannot be identical."
        )

    return True, ""


# ── Cross-field checks ──


def _check_timeline_order(data: dict[str, Any], fields: list[str]) -> tuple[bool, str]:
    """
    Cross-field check: ensure Declaration ≤ Ex-Div ≤ Record ≤ Payment.

    In US markets, ex-dividend date is typically 1 business day before
    record date, but the strict invariant is:
        declaration_date ≤ ex_dividend_date ≤ record_date ≤ payment_date
    """
    dates: dict[str, str] = {}
    for f in fields:
        v = data.get(f)
        if v is None:
            return True, ""  # Can't validate if any date is missing → skip
        dates[f] = str(v)

    # Check pairwise ordering
    pairs = [
        (fields[0], fields[1]),  # declaration ≤ ex-dividend
        (fields[1], fields[2]),  # ex-dividend ≤ record
        (fields[2], fields[3]),  # record ≤ payment
    ]
    for earlier, later in pairs:
        if dates[earlier] > dates[later]:
            return False, (
                f"[CRITICAL] Timeline integrity violation: '{earlier}'"
                f" ({dates[earlier]}) is AFTER '{later}' ({dates[later]})."
                f" Must satisfy: declaration_date ≤ ex_dividend_date ≤"
                f" record_date ≤ payment_date."
            )
    return True, ""


def _check_amount_requires_currency(
    data: dict[str, Any], amount_field: str, currency_field: str
) -> tuple[bool, str]:
    """
    Cross-field check: if amount is present, currency must also be present.

    In financial data, a dollar amount without a currency unit is meaningless.
    """
    amount = data.get(amount_field)
    currency = data.get(currency_field)

    if amount is not None and (currency is None or currency == ""):
        return False, (
            f"[HIGH] Amount-currency co-existence violation: '{amount_field}' ="
            f" {amount!r} but '{currency_field}' is missing or empty."
            f" A cash amount without a currency code is invalid financial data."
        )
    return True, ""


def _check_not_all_empty(data: dict[str, Any], fields: list[str]) -> tuple[bool, str]:
    """
    Cross-field check: at least one of the specified fields must be non‑empty.

    Prevents the LLM from classifying a document as Dividend while extracting
    no meaningful dividend data at all (category hallucination guard).
    """
    all_empty = True
    for f in fields:
        v = data.get(f)
        if v is not None and v != "" and v != 0.0:
            all_empty = False
            break

    if all_empty:
        field_list = ", ".join(fields)
        return False, (
            f"[CRITICAL] Empty action check failed: all dividend fields"
            f" ({field_list}) are empty/None. This suggests the document"
            f" may not actually be a dividend announcement, or the LLM"
            f" failed to extract any meaningful data."
        )
    return True, ""


# ── Dispatcher ──

_CHECK_DISPATCHER: dict[str, Any] = {
    "non_empty": _check_non_empty,
    "positive": _check_positive,
    "range": _check_range,
    "date_format": _check_date_format,
}


# ====================================================================
#  Rule executor
# ====================================================================


def _run_rule(rule: dict[str, Any], data: dict[str, Any]) -> tuple[bool, str]:
    """
    Execute a single validation rule against extracted data.

    Returns:
        (passed: bool, error_message: str)
    """
    field = rule["field"]
    check_name = rule["check"]

    # ── Cross-field check ──
    if check_name == "not_equal":
        return _check_not_equal(data, rule.get("fields", []))

    # ── Allowed-values check ──
    if check_name == "allowed_values":
        value = data.get(field)
        allowed = _ALLOWED_VALUES.get(field, set())
        passed = _check_allowed_values(value, allowed)
        if not passed:
            return False, (
                f"[{rule['severity']}] {rule['message']} "
                f"(field='{field}', value={value!r})"
            )
        return True, ""

    # ── Standard checks ──
    check_fn = _CHECK_DISPATCHER.get(check_name)
    if check_fn is None:
        return True, ""  # Unknown check — pass silently

    value = data.get(field)
    passed = check_fn(value)
    if not passed:
        return False, (
            f"[{rule['severity']}] {rule['message']} "
            f"(field='{field}', value={value!r})"
        )

    return True, ""


# ====================================================================
#  Node function
# ====================================================================


async def validator_node(state: PipelineState) -> dict:
    """
    LangGraph Node: validate extraction data against doc_type-specific rules.

    Reads ``extracted_data`` and ``doc_type`` from state, runs registered
    checks for that doc_type, and:
    - If all pass: sets ``validation_passed=True``
    - If any fail: sets ``validation_passed=False``, increments
      ``retry_count``, appends a ``CorrectionLog`` entry

    Args:
        state: PipelineState with ``extracted_data`` and ``doc_type`` populated.

    Returns:
        A dict of state updates for LangGraph to merge.
    """
    # ── Short-circuit: skip if a previous node already errored ──
    if state.get("error"):
        return {}

    extracted_data: dict[str, Any] | None = state.get("extracted_data")
    if not extracted_data:
        return {
            "validation_passed": False,
            "validation_report": (
                "validator_node: no extracted_data found in state."
            ),
            "error": "validator_node: no extracted_data found in state.",
        }

    # ── Resolve rules for this doc_type ──
    doc_type = state.get("doc_type", "")
    rules = _VALIDATION_RULESETS.get(doc_type)

    # No rules registered for this type → pass silently
    # (Pydantic schema already enforces structural validity)
    if rules is None:
        return {
            "validation_passed": True,
            "validation_report": (
                f"No validation rules registered for '{doc_type}' — "
                "passing by default."
            ),
        }

    # ── Run all validation rules ──
    errors: list[str] = []
    for rule in rules:
        passed, err_msg = _run_rule(rule, extracted_data)
        if not passed:
            errors.append(err_msg)

    # ── Assemble report ──
    if not errors:
        report = "All validation checks passed."
        new_validation_passed = True
    else:
        report = "Validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        new_validation_passed = False

    # ── Build state update ──
    updates: dict[str, Any] = {
        "validation_passed": new_validation_passed,
        "validation_report": report,
    }

    if not new_validation_passed:
        # ── Self-correction: increment retry count and log ──
        current_retry = state.get("retry_count", 0)
        new_retry = current_retry + 1

        new_log: CorrectionLog = {
            "cycle": new_retry,
            "error_summary": "; ".join(errors),
            "raw_feedback": report,
        }

        existing_logs: list[CorrectionLog] = state.get("correction_logs", [])
        updates["retry_count"] = new_retry
        updates["correction_logs"] = existing_logs + [new_log]

        # ── Check if we've exhausted retries ──
        max_retries = state.get("max_retries", 3)
        if new_retry >= max_retries:
            updates["error"] = (
                f"validator_node: max retries ({max_retries}) exhausted. "
                f"Document failed all correction cycles."
            )

    return updates
