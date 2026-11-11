"""
Golden-set runner (end-to-end extraction): run the pipeline on every golden
input and score it with ``evals/evaluators/golden.py``.

Dataset: ``evals/datasets/golden/`` (``inputs/`` + ``expected_pass/`` +
``expected_fail/``). Every case runs in a temporary database, raw store and
vector store, so evaluation never touches development data.

Usage:
    python evals/run_golden.py                       # eval all golden cases
    python evals/run_golden.py --threshold 0.9       # 90% pass threshold
    python evals/run_golden.py --json                # machine-readable JSON output

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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.evaluators.golden import (  # noqa: E402
    eval_expected_fail,
    eval_expected_pass,
    load_all_golden,
)
from src.config import settings  # noqa: E402
from src.rag.chroma_client import override_chroma_path  # noqa: E402
from src.workflows.graph import run_pipeline  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "datasets" / "golden"
RAW_DATA_DIR = GOLDEN_DIR / "inputs"


# ── Pipeline execution ───────────────────────────────────────────────


def _run_pipeline_for(source_file: str, raw_data_dir: Path = RAW_DATA_DIR) -> dict:
    """
    Run the pipeline on one source file inside a temp directory so eval
    does NOT pollute the development database or vector store.
    """
    src_path = Path(raw_data_dir) / source_file
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with tempfile.TemporaryDirectory(prefix="pipeline_eval_") as tmpdir:
        # Override DB paths so eval runs in complete isolation
        original_db = settings.sqlite_db_path
        original_chroma = settings.chromadb_path
        original_raw = settings.raw_store_path
        settings.sqlite_db_path = str(Path(tmpdir) / "eval.db")
        settings.chromadb_path = str(Path(tmpdir) / "chromadb")
        settings.raw_store_path = str(Path(tmpdir) / "raw")

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
            settings.raw_store_path = original_raw
            # Reset ChromaDB singleton back to the production path
            override_chroma_path(None)

    return state


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

    goldens = load_all_golden(GOLDEN_DIR)
    if not goldens:
        print(f"No golden files found in {GOLDEN_DIR}.")
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
            state = _run_pipeline_for(source_file, raw_data_dir)
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
            passed, total, details = eval_expected_pass(golden, state)
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
            passed, total, details = eval_expected_fail(golden, state)
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
