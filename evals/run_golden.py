"""
Golden-set runner: run the pipeline end to end on every golden case and gate
on the result (scoring: ``evals/evaluators/golden.py``).

Dataset: ``evals/datasets/golden/`` (``cases/*.json`` + ``inputs/``). Every case
runs in a temporary database, raw store and vector store, so evaluation never
touches development data.

Usage:
    python evals/run_golden.py              # run, print, save results/golden/<ts>/
    python evals/run_golden.py --json       # also print the summary as JSON
    python evals/run_golden.py --no-save    # CI: print only

Exit code:
    0 — gate passed: every gating case reached its expected outcome and its
        REQUIRED fields, and no run was inconclusive
    1 — gate failed, a run was inconclusive, or no cases were found
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.evaluators.golden import (  # noqa: E402
    IMPLEMENTED_FEATURES,
    GoldenCase,
    load_cases,
    score_case,
    summarize,
)
from src.config import settings  # noqa: E402
from src.rag.chroma_client import override_chroma_path  # noqa: E402
from src.workflows.graph import run_pipeline  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "datasets" / "golden"
RESULTS_DIR = REPO_ROOT / "results" / "golden"


# ── Pipeline execution ───────────────────────────────────────────────


def _run_pipeline_for(relative_input: str, base_dir: Path = GOLDEN_DIR) -> dict:
    """
    Run the pipeline on one input inside a temp directory so evaluation
    does NOT pollute the development database or vector store.
    """
    src_path = Path(base_dir) / relative_input
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with tempfile.TemporaryDirectory(prefix="pipeline_eval_") as tmpdir:
        original = (
            settings.sqlite_db_path,
            settings.chromadb_path,
            settings.raw_store_path,
        )
        settings.sqlite_db_path = str(Path(tmpdir) / "eval.db")
        settings.chromadb_path = str(Path(tmpdir) / "chromadb")
        settings.raw_store_path = str(Path(tmpdir) / "raw")
        override_chroma_path(settings.chromadb_path)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state = loop.run_until_complete(run_pipeline(str(src_path)))
        finally:
            loop.close()
            (
                settings.sqlite_db_path,
                settings.chromadb_path,
                settings.raw_store_path,
            ) = original
            override_chroma_path(None)
    return state


def _tokens(state: dict) -> dict[str, int]:
    steps = state.get("step_runs") or []
    return {
        "llm_calls": sum(1 for s in steps if s.get("prompt_tokens") is not None),
        "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in steps),
        "completion_tokens": sum(s.get("completion_tokens") or 0 for s in steps),
    }


def run_case(case: GoldenCase, base_dir: Path = GOLDEN_DIR) -> dict:
    """Run and score one case; a crash is reported as a failed case."""
    try:
        state = _run_pipeline_for(case.input, base_dir)
    except Exception as exc:  # the gate must report, not crash
        state = {
            "error": f"runner: {exc}",
            "failure": {
                "stage": "runner",
                "code": "RUNNER_ERROR",
                "category": "unknown",
            },
        }
    result = score_case(case, state)
    result["tokens"] = _tokens(state)
    return result


# ── Reporting ────────────────────────────────────────────────────────


def _print_case(r: dict) -> None:
    tag = r["status"].upper() if r["gating"] else f"PENDING:{r['status']}"
    print(
        f"  [{tag}] {r['case_id']}: expected {r['expected_outcome']}, "
        f"got {r['actual_outcome']}"
        + (f"  (failure {r['failure']})" if r["failure"] else "")
    )
    for c in r["checks"]:
        if not c["passed"]:
            print(
                f"      XX {c['check']}: expected {c['expected']!r}, got {c['actual']!r}"
            )
    for f in r["fields"]:
        if not f["passed"]:
            level = "REQUIRED" if f["required"] else "warning"
            print(f"      XX {f['field']} ({level}): {f['detail']}")
    if not r["gating"]:
        print(f"      pending on: {', '.join(r['pending_features'])}")


def _print_summary(s: dict) -> None:
    print(f"\n{'=' * 60}")
    print(
        f"Outcome gate: {s['outcome_correct']}/{s['outcome_scored']} gating cases "
        f"reached the expected outcome"
    )
    if s["false_accepts"]:
        print(f"  FALSE ACCEPTS (stored but should not be): {s['false_accepts']}")
    for f, c in sorted(s["field_accuracy"].items()):
        print(f"  field {f:32s} {c['passed']}/{c['total']}")
    if s["pending"]:
        print(f"Pending (reported, not gating): {s['pending']}")
    if s["pending_now_passing"]:
        print(
            f"  Pending cases now passing; consider gating: {s['pending_now_passing']}"
        )
    if s["inconclusive"]:
        print(f"INCONCLUSIVE (infrastructure): {s['inconclusive']}")
    verdict = "PASS" if s["gate_passed"] else "FAIL"
    print(f"GATE: {verdict}" + (f"  failed: {s['failed']}" if s["failed"] else ""))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()


def _save(results: list[dict], summary: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / stamp
    out.mkdir(parents=True)
    manifest = {
        "created_at_utc": stamp,
        "model": settings.deepseek_model,
        "implemented_features": sorted(IMPLEMENTED_FEATURES),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "tokens": {
            k: sum(r["tokens"][k] for r in results)
            for k in ("llm_calls", "prompt_tokens", "completion_tokens")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "per_case.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return out


# ── Main entry point ─────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Golden set: outcome gate + field gate.")
    p.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    p.add_argument(
        "--no-save", action="store_true", help="Do not write results/golden/."
    )
    p.add_argument(
        "--golden-dir",
        type=Path,
        default=GOLDEN_DIR,
        help=f"Golden dataset directory. Default: {GOLDEN_DIR}",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cases = load_cases(args.golden_dir)
    if not cases:
        print(f"No golden cases found in {args.golden_dir / 'cases'}.")
        return 1

    results = []
    for case in cases:
        r = run_case(case, args.golden_dir)
        _print_case(r)
        results.append(r)

    summary = summarize(results)
    _print_summary(summary)
    if not args.no_save:
        print(f"saved: {_save(results, summary).relative_to(REPO_ROOT)}")
    if args.json:
        print("\n--- Machine-readable JSON ---")
        print(json.dumps(summary, indent=2))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
