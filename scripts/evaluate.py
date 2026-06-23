"""
Phase 3-C: Golden Dataset Eval Runner

Compares pipeline output against golden (hand-labelled) extraction results.
Supports field-type-aware comparison with per-field eval overrides.

Usage:
    python scripts/evaluate.py                              # eval all golden cases
    python scripts/evaluate.py --threshold 0.9              # 90% pass threshold
    python scripts/evaluate.py --json                       # machine-readable JSON output

Exit code:
    0 — accuracy >= threshold (PASS)
    1 — accuracy <  threshold (FAIL) or error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.config import settings
from src.graph.builder import run_pipeline
from src.rag.chroma_client import override_chroma_path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "golden"
RAW_DATA_DIR = REPO_ROOT / "raw_data" / "sample"

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


def _compare_field(
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


def _load_all_golden() -> dict[str, dict]:
    """Scan golden/ directory and return {source_file: golden_data}."""
    goldens: dict[str, dict] = {}
    for subdir in ("expected_pass", "expected_fail"):
        path = GOLDEN_DIR / subdir
        if not path.exists():
            continue
        for fpath in sorted(path.glob("*.golden.json")):
            with open(fpath) as f:
                data = json.load(f)
            src = data.get("source_file", fpath.stem.replace(".golden", ""))
            goldens[src] = data
    return goldens


# ── Pipeline execution ───────────────────────────────────────────────


def _run_pipeline_for(source_file: str) -> dict:
    """
    Run the pipeline on one source file inside a temp directory so eval
    does NOT pollute the development database or vector store.
    """
    src_path = RAW_DATA_DIR / source_file
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with tempfile.TemporaryDirectory(prefix="pipeline_eval_") as tmpdir:
        # Override DB paths so eval runs in complete isolation
        original_db = settings.sqlite_db_path
        original_chroma = settings.chromadb_path
        settings.sqlite_db_path = str(Path(tmpdir) / "eval.db")
        settings.chromadb_path = str(Path(tmpdir) / "chromadb")

        # Reset ChromaDB singleton so the next get_chroma_client() call
        # picks up the temp path. Same pattern used in test_rag.py fixtures.
        override_chroma_path(settings.chromadb_path)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state = loop.run_until_complete(run_pipeline(str(src_path)))
        finally:
            loop.close()
            # Restore original settings
            settings.sqlite_db_path = original_db
            settings.chromadb_path = original_chroma
            # Reset ChromaDB singleton back to the production path
            override_chroma_path(None)

    return state


def _extract_actual_fields(state: dict) -> dict:
    """Extract business fields from PipelineState, stripping pipeline metadata."""
    extracted = state.get("extracted_data") or {}
    return {k: v for k, v in extracted.items() if k not in _PIPELINE_FIELDS}


# ── Evaluation logic ─────────────────────────────────────────────────


def _eval_expected_pass(
    golden: dict,
    state: dict,
) -> tuple[int, int, list[dict]]:
    """
    Evaluate one expected_pass case.
    Returns (passed_count, total_count, details).
    """
    golden_fields: dict = golden.get("fields", {})
    overrides: dict = golden.get("_field_eval_overrides", {})
    actual_fields: dict = _extract_actual_fields(state)

    passed = 0
    total = len(golden_fields)
    details: list[dict] = []

    for field_name, expected_val in golden_fields.items():
        actual_val = actual_fields.get(field_name)
        field_overrides = overrides.get(field_name)
        ok, detail = _compare_field(
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


def _eval_expected_fail(golden: dict, state: dict) -> tuple[int, int, list[dict]]:
    """
    Evaluate one expected_fail case.
    Only checks validation_passed == false. No field-level comparison.
    """
    validation_passed = state.get("validation_passed")
    error = state.get("error")

    if error:
        # Fatal error means the document failed to process at all — acceptable
        return (
            1,
            1,
            [
                {
                    "check": "validation_passed",
                    "passed": True,
                    "detail": f"error: {error}",
                }
            ],
        )

    if validation_passed is False:
        return (
            1,
            1,
            [
                {
                    "check": "validation_passed",
                    "passed": True,
                    "detail": "validation_passed=false",
                }
            ],
        )
    if validation_passed is True:
        return (
            0,
            1,
            [
                {
                    "check": "validation_passed",
                    "passed": False,
                    "detail": "expected fail but validation passed",
                }
            ],
        )

    # validation_passed is None (pipeline didn't reach validator)
    return (
        0,
        1,
        [
            {
                "check": "validation_passed",
                "passed": False,
                "detail": "validation_passed is None",
            }
        ],
    )


# ── Main entry point ─────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Golden Dataset Evaluation — compare pipeline output against hand-labelled golden files."
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Minimum accuracy threshold (0.0–1.0). Exit 1 if below. Default: 0.85",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable).",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Directory containing source .txt files. Default: {RAW_DATA_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    goldens = _load_all_golden()
    if not goldens:
        print("No golden files found in golden/ directory.")
        return 1

    total_passed = 0
    total_checks = 0
    results: list[dict] = []
    raw_data_dir = Path(args.raw_data_dir)

    for source_file, golden in sorted(goldens.items()):
        src_path = raw_data_dir / source_file
        if not src_path.exists():
            results.append(
                {
                    "source_file": source_file,
                    "skipped": True,
                    "reason": f"source file not found: {src_path}",
                }
            )
            print(f"  SKIP {source_file}: source file not found")
            continue

        # Run pipeline
        try:
            state = _run_pipeline_for(source_file)
        except Exception as e:
            results.append(
                {
                    "source_file": source_file,
                    "error": str(e),
                }
            )
            print(f"  FAIL {source_file}: pipeline error — {e}")
            continue

        # Determine if expected pass or fail
        expected_pass = golden.get("validation_passed") is True

        if expected_pass:
            passed, total, details = _eval_expected_pass(golden, state)
            pass_rate = passed / total if total > 0 else 1.0
            total_passed += passed
            total_checks += total

            verdict = "PASS" if pass_rate >= args.threshold else "FAIL"
            print(
                f"  [{verdict}] {source_file}: "
                f"{passed}/{total} fields passed ({pass_rate:.0%})"
            )
            for d in details:
                flag = "OK" if d["passed"] else "XX"
                print(f"      [{flag}] {d['field']}: {d['detail']}")
                if not d["passed"]:
                    print(f"             actual  : {d['actual']}")
                    print(f"             expected: {d['expected']}")

            results.append(
                {
                    "source_file": source_file,
                    "type": "expected_pass",
                    "passed": passed,
                    "total": total,
                    "pass_rate": pass_rate,
                    "details": details,
                }
            )
        else:
            passed, total, details = _eval_expected_fail(golden, state)
            total_passed += passed
            total_checks += total

            verdict = "PASS" if passed == total else "FAIL"
            print(
                f"  [{verdict}] {source_file} (expected_fail): {passed}/{total} checks passed"
            )
            for d in details:
                flag = "OK" if d["passed"] else "XX"
                print(f"      [{flag}] {d['check']}: {d['detail']}")

            results.append(
                {
                    "source_file": source_file,
                    "type": "expected_fail",
                    "passed": passed,
                    "total": total,
                    "pass_rate": passed / total if total > 0 else 0.0,
                    "details": details,
                }
            )

    # ── Summary ──
    overall_rate = total_passed / total_checks if total_checks > 0 else 0.0
    passed_threshold = overall_rate >= args.threshold

    summary = {
        "total_passed": total_passed,
        "total_checks": total_checks,
        "overall_accuracy": round(overall_rate, 4),
        "threshold": args.threshold,
        "passed": passed_threshold,
        "results": results,
    }

    print(f"\n{'=' * 56}")
    if passed_threshold:
        print(
            f"PASS: {total_passed}/{total_checks} "
            f"({overall_rate:.1%}) — threshold >= {args.threshold:.0%}"
        )
    else:
        print(
            f"FAIL: {total_passed}/{total_checks} "
            f"({overall_rate:.1%}) — threshold < {args.threshold:.0%}"
        )

    if args.json:
        print("\n--- Machine-readable JSON ---")
        print(json.dumps(summary, indent=2))

    return 0 if passed_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
