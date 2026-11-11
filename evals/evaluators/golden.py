"""
Golden-set evaluator: score one pipeline run against a hand-labelled golden file.

Pure scoring (no LLM, no I/O besides loading golden files). The runner
``evals/run_golden.py`` executes the pipeline and calls these functions.

Two kinds of case (``evals/datasets/golden/``):
  * ``expected_pass`` — every labelled field must match, compared by field type
    (``enum``, ``numeric`` ±1 %, ``bool``, ``date``, ``text_lower``), with
    per-field overrides ``date_range``, ``fuzzy_match`` and ``accept_null``
  * ``expected_fail`` — the pipeline must *judge and reject* the document; an
    infrastructure error (LLM unreachable, missing file) earns no credit
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.nodes.extractor import EXTRACTOR_MODELS

# ── Field-type taxonomy ──────────────────────────────────────────────
# Maps field name -> comparison mode for M&A and Dividend schemas.
# Fallback is "text_lower" (case-insensitive exact match).

FIELD_TYPE_MAP: dict[str, str] = {
    # Enum fields (must match golden exactly)
    "payment_method": "enum",
    "currency": "enum",
    "dividend_type": "enum",
    "frequency": "enum",
    # Numeric fields (relative error tolerance)
    "total_value_usd": "numeric",
    "stake_percentage": "numeric",
    "dividend_cash_amount": "numeric",
    # Boolean fields (exact match)
    "requires_shareholder_approval": "bool",
    # Date fields (exact match YYYY-MM-DD, but may have overrides)
    "announcement_date": "date",
    "expected_close_date": "date",
    "declaration_date": "date",
    "record_date": "date",
    "ex_dividend_date": "date",
    "payment_date": "date",
    # Text fields (case-insensitive exact match)
    "acquirer": "text_lower",
    "target": "text_lower",
    "ticker": "text_lower",
}

NUMERIC_RELATIVE_TOLERANCE = 0.01  # 1%

# Failure categories that say nothing about extraction quality.
_INFRASTRUCTURE_CATEGORIES = {"transient", "input_error"}

# Pipeline-controlled fields to strip from extracted_data
_PIPELINE_FIELDS = {"doc_id", "metadata", "raw_text_snippet"}


# ── Comparison helpers ───────────────────────────────────────────────


def compare_field(
    field_name: str,
    actual: Any,
    expected: Any,
    overrides: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Compare a single field; return (passed, detail_message)."""
    # Handle null/None cases first
    if actual is None and expected is None:
        return True, "both null"
    if actual is None and expected is not None:
        if overrides and overrides.get("accept_null"):
            return True, "null accepted via override"
        return False, f"expected '{expected}', got null"

    # ── Check for eval overrides ──
    if overrides:
        override_mode = overrides.get("mode")

        if override_mode == "date_range":
            _start = overrides.get("range_start")
            _end = overrides.get("range_end")
            if not _start or not _end:
                return False, "date_range override missing range_start/range_end"
            if actual is None and overrides.get("accept_null"):
                return True, "null accepted via date_range override"
            if isinstance(actual, str) and _start <= actual <= _end:
                return True, f"date '{actual}' in range [{_start}, {_end}]"
            return False, f"date '{actual}' outside range [{_start}, {_end}]"

        if override_mode == "fuzzy_match":
            threshold = overrides.get("threshold", 0.85)
            ratio = SequenceMatcher(
                None,
                str(actual).lower().strip(),
                str(expected).lower().strip(),
            ).ratio()
            if ratio >= threshold:
                return True, f"fuzzy match ratio={ratio:.3f} >= {threshold}"
            return False, f"fuzzy match ratio={ratio:.3f} < {threshold}"

    # ── Mode-based comparison ──
    mode = FIELD_TYPE_MAP.get(field_name, "text_lower")

    if mode == "enum":
        if str(actual).strip() == str(expected).strip():
            return True, "exact match"
        return False, f"expected enum '{expected}', got '{actual}'"

    if mode == "numeric":
        try:
            actual_f = float(actual)
            expected_f = float(expected)
        except (ValueError, TypeError):
            return (
                False,
                f"numeric comparison failed: actual='{actual}', expected='{expected}'",
            )
        if expected_f == 0.0:
            rel_err = abs(actual_f)
            passed = rel_err < 1e-6
        else:
            rel_err = abs(actual_f - expected_f) / abs(expected_f)
            passed = rel_err <= NUMERIC_RELATIVE_TOLERANCE
        detail = (
            "within 1% tolerance"
            if passed
            else f"actual={actual_f}, expected={expected_f}, relative_error={rel_err:.4f}"
        )
        return passed, detail

    if mode == "bool":
        actual_b = bool(actual) if not isinstance(actual, bool) else actual
        expected_b = bool(expected) if not isinstance(expected, bool) else expected
        if actual_b == expected_b:
            return True, "exact match"
        return False, f"expected bool '{expected_b}', got '{actual_b}'"

    if mode in ("date", "text_lower"):
        actual_s = str(actual).strip().lower()
        expected_s = str(expected).strip().lower()
        if actual_s == expected_s:
            return True, "exact match (case-insensitive)"
        return False, f"expected '{expected}', got '{actual}'"

    # Fallback: exact match
    if str(actual).strip() == str(expected).strip():
        return True, "exact match"
    return False, f"expected '{expected}', got '{actual}'"


# ── Golden loading ───────────────────────────────────────────────────


def load_all_golden(golden_dir: Path) -> dict[str, dict]:
    """Scan ``golden_dir`` and return {source_file: golden_data}."""
    goldens: dict[str, dict] = {}
    for subdir in ("expected_pass", "expected_fail"):
        path = Path(golden_dir) / subdir
        if not path.exists():
            continue
        for fpath in sorted(path.glob("*.golden.json")):
            with open(fpath) as f:
                data = json.load(f)
            src = data.get("source_file", fpath.stem.replace(".golden", ""))
            goldens[src] = data
    return goldens


def extract_actual_fields(state: dict) -> dict:
    """Extract business fields from PipelineState, stripping pipeline metadata."""
    extracted = state.get("extracted_data") or {}
    return {k: v for k, v in extracted.items() if k not in _PIPELINE_FIELDS}


# ── Scoring ──────────────────────────────────────────────────────────


def eval_expected_pass(
    golden: dict,
    state: dict,
) -> tuple[int, int, list[dict]]:
    """
    Evaluate one expected_pass case.
    Returns (passed_count, total_count, details).
    """
    golden_fields: dict = golden.get("fields", {})
    overrides: dict = golden.get("_field_eval_overrides", {})
    actual_fields: dict = extract_actual_fields(state)

    passed = 0
    total = len(golden_fields)
    details: list[dict] = []

    for field_name, expected_val in golden_fields.items():
        actual_val = actual_fields.get(field_name)
        field_overrides = overrides.get(field_name)
        ok, detail = compare_field(
            field_name, actual_val, expected_val, field_overrides
        )
        if ok:
            passed += 1
        details.append(
            {
                "field": field_name,
                "passed": ok,
                "detail": detail,
                "actual": actual_val,
                "expected": expected_val,
            }
        )

    return passed, total, details


def is_legitimate_rejection(state: dict) -> bool:
    """
    True when the pipeline *judged* the document and refused it, as opposed
    to merely crashing.

    Legitimate rejections:
      - the Extraction Validator (or the Extractor's schema-parse loop) marked
        the document ``validation_passed=False`` after exhausting retries;
      - the Classifier succeeded and labelled the document ``none`` / ``unclear``
        (or any type the pipeline has no extractor for);
      - a classification or extraction loop ran out of retries on output that
        never passed validation.

    Anything else carrying an ``error`` (LLM unreachable, bad API key,
    missing file, ...) is an infrastructure failure and proves nothing
    about extraction quality, so it must NOT earn credit.
    """
    if state.get("validation_passed") is False:
        return True
    failure = state.get("failure") or {}
    if failure.get("category") in _INFRASTRUCTURE_CATEGORIES:
        return False
    if failure:
        # Any other structured failure means the pipeline judged the document
        # (out of scope, ambiguous, or the classifier/extractor output never
        # passed validation) rather than hit an infrastructure problem.
        return True
    doc_type = state.get("doc_type")
    return doc_type is not None and doc_type not in EXTRACTOR_MODELS


def eval_expected_fail(golden: dict, state: dict) -> tuple[int, int, list[dict]]:
    """
    Evaluate one expected_fail case.

    Passes only if the pipeline legitimately rejected the document (see
    ``is_legitimate_rejection``); an infrastructure error is a FAIL.
    """
    error = state.get("error")

    if state.get("validation_passed") is True:
        passed, detail = False, "expected fail but validation passed"
    elif is_legitimate_rejection(state):
        reason = f"error: {error}" if error else "validation_passed=false"
        if state.get("validation_passed") is not False:
            reason = (
                f"classified as '{state.get('doc_type')}' (no extractor) — {reason}"
            )
        passed, detail = True, reason
    elif error:
        passed, detail = False, f"infrastructure error, not a rejection: {error}"
    else:
        passed, detail = False, "validation_passed is None"

    return (
        1 if passed else 0,
        1,
        [{"check": "validation_passed", "passed": passed, "detail": detail}],
    )
