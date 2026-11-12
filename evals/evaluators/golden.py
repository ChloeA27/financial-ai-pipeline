"""
Golden-set evaluator: one end-to-end case per pipeline path, scored by outcome.

Pure scoring (no LLM, no I/O besides loading case files). The runner
``evals/run_golden.py`` executes the pipeline and calls these functions.

Each case (``evals/datasets/golden/cases/<id>.json``) states the outcome the
pipeline must reach: ``stored``, ``out_of_scope``, ``human_review`` or ``dlq``
(see ``src/schemas/outcome.py``). It also gives the failure stage / category
for a case that is not stored, and the labelled fields for one that is.

Two gates, never pooled:
  * outcome gate: every gating case reaches its expected outcome (and failure
    stage / category / code where given). One miss fails the gate.
  * field gate: on cases correctly stored, every REQUIRED field (from the
    document type's ValidationProfile) matches. Other fields are reported as
    accuracy and warnings only.

A case whose ``requires`` names a feature not in ``IMPLEMENTED_FEATURES`` is
run and reported as ``pending`` but cannot fail the gate (like pytest xfail).
A run stopped by an infrastructure failure is ``inconclusive``: it fails the
gate and is excluded from every accuracy number.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from src.nodes.extractor import EXTRACTOR_MODELS
from src.schemas.failure import CAT_TRANSIENT
from src.schemas.outcome import OUTCOME_STORED, OUTCOMES, outcome_of

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


# ── Cases ────────────────────────────────────────────────────────────

# Features a case may depend on via ``requires``. Add a name here when the
# pipeline implements it; its cases then gate automatically.
#   presence_levels — OPTIONAL fields may be empty (the validator reads the profile)
#   retry_policy    — whether to retry depends on the failure category
IMPLEMENTED_FEATURES: frozenset[str] = frozenset()

# Infrastructure failures say nothing about the pipeline's judgement.
INCONCLUSIVE_CATEGORIES = frozenset({CAT_TRANSIENT, "config_error"})


@dataclass
class GoldenCase:
    case_id: str
    input: str
    purpose: str
    expected: dict[str, Any]
    synthetic: bool = True
    requires: list[str] = field(default_factory=list)

    @property
    def missing_features(self) -> list[str]:
        return [f for f in self.requires if f not in IMPLEMENTED_FEATURES]


def load_cases(golden_dir: Path) -> list[GoldenCase]:
    """Read every ``cases/*.json`` under ``golden_dir``; validate the outcome."""
    cases = []
    for path in sorted((Path(golden_dir) / "cases").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        outcome = data["expected"]["outcome"]
        if outcome not in OUTCOMES:
            raise ValueError(f"{path.name}: unknown outcome {outcome!r}")
        cases.append(
            GoldenCase(
                case_id=data["case_id"],
                input=data["input"],
                purpose=data.get("purpose", ""),
                expected=data["expected"],
                synthetic=bool(data.get("synthetic", True)),
                requires=list(data.get("requires") or []),
            )
        )
    return cases


def extract_actual_fields(state: dict) -> dict:
    """Business fields from the final state, without pipeline metadata."""
    extracted = state.get("extracted_data") or {}
    return {k: v for k, v in extracted.items() if k not in _PIPELINE_FIELDS}


def required_fields(doc_type: Optional[str]) -> frozenset[str]:
    """REQUIRED fields of a document type, from its ValidationProfile."""
    model = EXTRACTOR_MODELS.get(doc_type or "")
    profile = getattr(model, "ValidationProfile", None)
    return getattr(profile, "REQUIRED_FIELDS", frozenset())


# ── Scoring ──────────────────────────────────────────────────────────


def _check(name: str, expected: Any, actual: Any) -> dict:
    return {
        "check": name,
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
    }


def score_case(case: GoldenCase, state: dict) -> dict:
    """
    Score one run.

    ``status`` is ``pass`` | ``fail`` | ``inconclusive``; ``gating`` says
    whether the case counts toward the gate (False while features are pending).
    """
    exp = case.expected
    failure = state.get("failure") or {}
    actual_outcome = outcome_of(state)

    checks = [_check("outcome", exp["outcome"], actual_outcome)]
    if "doc_type" in exp:
        checks.append(_check("doc_type", exp["doc_type"], state.get("doc_type")))
    for key in ("stage", "category", "code"):
        if key in (exp.get("failure") or {}):
            checks.append(
                _check(f"failure.{key}", exp["failure"][key], failure.get(key))
            )

    fields: list[dict] = []
    if exp["outcome"] == OUTCOME_STORED and actual_outcome == OUTCOME_STORED:
        required = required_fields(exp.get("doc_type"))
        overrides = exp.get("field_overrides") or {}
        actual_fields = extract_actual_fields(state)
        for name, expected_val in (exp.get("fields") or {}).items():
            actual_val = actual_fields.get(name)
            ok, detail = compare_field(
                name, actual_val, expected_val, overrides.get(name)
            )
            fields.append(
                {
                    "field": name,
                    "required": name in required,
                    "passed": ok,
                    "detail": detail,
                    "actual": actual_val,
                    "expected": expected_val,
                }
            )

    if failure.get("category") in INCONCLUSIVE_CATEGORIES:
        status = "inconclusive"
    elif all(c["passed"] for c in checks) and all(
        f["passed"] for f in fields if f["required"]
    ):
        status = "pass"
    else:
        status = "fail"

    return {
        "case_id": case.case_id,
        "input": case.input,
        "status": status,
        "gating": not case.missing_features,
        "pending_features": case.missing_features,
        "expected_outcome": exp["outcome"],
        "actual_outcome": actual_outcome,
        "false_accept": exp["outcome"] != OUTCOME_STORED
        and actual_outcome == OUTCOME_STORED,
        "failure": (
            {k: failure.get(k) for k in ("stage", "code", "category")}
            if failure
            else None
        ),
        "checks": checks,
        "fields": fields,
        "field_warnings": [
            f["field"] for f in fields if not f["passed"] and not f["required"]
        ],
    }


def summarize(results: list[dict]) -> dict:
    """Gate verdict plus the numbers behind it (the two gates are never pooled)."""
    gating = [r for r in results if r["gating"]]
    pending = [r for r in results if not r["gating"]]
    failed = [r["case_id"] for r in gating if r["status"] == "fail"]
    inconclusive = [r["case_id"] for r in results if r["status"] == "inconclusive"]
    scored = [r for r in gating if r["status"] != "inconclusive"]

    confusion = Counter(
        (r["expected_outcome"], r["actual_outcome"])
        for r in results
        if r["status"] != "inconclusive"
    )
    per_field: dict[str, dict[str, int]] = {}
    for r in scored:
        for f in r["fields"]:
            c = per_field.setdefault(f["field"], {"passed": 0, "total": 0})
            c["total"] += 1
            c["passed"] += int(f["passed"])

    return {
        "gate_passed": not failed and not inconclusive,
        "cases": len(results),
        "gating_cases": len(gating),
        "outcome_correct": sum(
            r["expected_outcome"] == r["actual_outcome"] for r in scored
        ),
        "outcome_scored": len(scored),
        "failed": failed,
        "inconclusive": inconclusive,
        "false_accepts": [r["case_id"] for r in results if r["false_accept"]],
        "pending": {r["case_id"]: r["pending_features"] for r in pending},
        "pending_now_passing": [r["case_id"] for r in pending if r["status"] == "pass"],
        "confusion": [
            {"expected": e, "actual": a, "count": n}
            for (e, a), n in sorted(confusion.items())
        ],
        "field_accuracy": per_field,
        "field_warnings": {
            r["case_id"]: r["field_warnings"] for r in scored if r["field_warnings"]
        },
    }
