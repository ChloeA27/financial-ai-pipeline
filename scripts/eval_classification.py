"""
E1 — classification-stage evaluation runner (offline paired comparison).

Runs the real classification path (Reader → Classifier ⇄ Classification
Validator, see ``build_classification_graph``) on a frozen list of E1 cases,
once per input strategy (variant) and repetition, and writes reproducible
results under ``results/e1/<timestamp>_<tag>/``.

Modes:
  --coverage-only   no LLM calls: input coverage per strategy (is the gold
                    evidence inside the window?)
  --estimate        no LLM calls: token estimate for the planned run
  (default)         the real run; stops at --max-calls

Usage::

    python scripts/eval_classification.py --dataset <E1 dir>/e1_dataset_draft.jsonl \\
        --coverage-only
    python scripts/eval_classification.py --dataset ... --estimate
    python scripts/eval_classification.py --dataset ... --variants head,structured,keyword \\
        --reps 2 --temperature 0 --max-calls 400

The test split is refused unless --allow-test is given: it must not be run
before the prompt and input strategy are final.

Blind-labelling protection: per-case files only show gold / draft labels for
human-verified cases.  Unverified (draft) cases appear in aggregates only, so
opening the results cannot anchor a labeller on Claude's sealed drafts.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import settings  # noqa: E402
from src.evaluation import e1  # noqa: E402
from src.nodes.input_window import (  # noqa: E402
    CHARS_PER_TOKEN,
    STRATEGIES,
    build_input,
)
from src.nodes.reader import SourceDecodeError, decode_source  # noqa: E402

CLASSIFY_OUTPUT_TOKENS_EST = 150


# ── Helpers ──────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _system_message_chars() -> int:
    from src.llm.client import _build_schema_hint
    from src.nodes.classifier import SYSTEM_PROMPT, ClassificationResult

    return len(SYSTEM_PROMPT) + 200 + len(_build_schema_hint(ClassificationResult))


def _windows(text: str, budget_tokens: int) -> dict[str, str]:
    """Every strategy's window, plus head at the same budget (to separate budget from strategy)."""
    out = {s: build_input(text, s, budget_tokens).text for s in STRATEGIES}
    out["head_same_budget"] = text[: budget_tokens * CHARS_PER_TOKEN]
    return out


def _out_dir(base: Path, tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = base / f"{stamp}_{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _manifest(args: argparse.Namespace, cases: list[e1.Case], mode: str) -> dict:
    return {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset.name,  # file name only: no local paths in a public repo
        "dataset_sha256": _sha256_file(args.dataset),
        "split": args.split,
        "guide_versions": sorted({c.guide_version or "?" for c in cases}),
        "projection_rule": e1.PROJECTION_RULE,
        "model": settings.deepseek_model,
        "temperature": args.temperature,
        "budget_tokens": args.budget_tokens,
        "variants": args.variants,
        "reps": args.reps,
        "git": _git_state(),
        "cases": [
            {
                "case_id": c.case_id,
                "group": c.group,
                "verified": c.verified,
                "input_sha256": _sha256_file(c.input_file),
            }
            for c in cases
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_text(case: e1.Case) -> Optional[str]:
    try:
        return decode_source(case.input_file.read_bytes())[0]
    except SourceDecodeError:
        return None


# ── Coverage-only (no LLM) ───────────────────────────────────────────


def run_coverage(args: argparse.Namespace, cases: list[e1.Case]) -> Path:
    out = _out_dir(args.out, f"coverage_{args.tag}")
    rows = []
    for case in cases:
        text = _read_text(case)
        if text is None:
            continue
        windows = _windows(text, args.budget_tokens)
        row = {
            "case_id": case.case_id,
            "group": case.group,
            "verified": case.verified,
            "gold_projected": "+".join(e1.project(case.labels)),
            "chars_total": len(text),
        }
        for arm, win in windows.items():
            cov = e1.coverage_for(case, win)
            row[f"cov_{arm}"] = "" if cov is None else int(cov)
            row[f"chars_{arm}"] = len(win)
        rows.append(row)
    arms = list(_windows("", args.budget_tokens).keys())
    _write_csv(out / "coverage.csv", [_redact_coverage(r, arms) for r in rows])
    summary: dict[str, Any] = {}
    for scope in ("verified_positive", "all_positive_drafts"):
        sel = [
            r
            for r in rows
            if r["cov_head"] != "" and (scope == "all_positive_drafts" or r["verified"])
        ]
        summary[scope] = {"n": len(sel)}
        for arm in arms:
            summary[scope][arm] = e1._rate(sum(r[f"cov_{arm}"] for r in sel), len(sel))
        by_group: dict[str, dict] = {}
        for g in sorted({r["group"] for r in sel}):
            gs = [r for r in sel if r["group"] == g]
            by_group[g] = {
                "n": len(gs),
                **{
                    arm: e1._rate(sum(r[f"cov_{arm}"] for r in gs), len(gs))
                    for arm in arms
                },
            }
        summary[scope]["by_group"] = by_group
    (out / "manifest.json").write_text(
        json.dumps(_manifest(args, cases, "coverage-only"), indent=2), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_coverage(summary, arms)
    print(f"\nWritten to {out}")
    return out


def _redact_coverage(row: dict[str, Any], arms: list[str]) -> dict[str, Any]:
    """Hide draft labels (and coverage, which reveals them) for unverified cases."""
    if row["verified"]:
        return row
    hidden = {"gold_projected": ""}
    hidden.update({f"cov_{a}": "" for a in arms})
    return {**row, **hidden}


def _redact_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = {**rec, "gold_projected": "+".join(rec["gold_projected"])}
    if not rec["verified"]:
        out.update(gold_projected="", expected="", correct="", multi="", coverage="")
    return out


def _print_coverage(summary: dict, arms: list[str]) -> None:
    for scope, s in summary.items():
        print(f"\nInput coverage — {scope} (n={s['n']})")
        print("  " + "group".ljust(12) + "".join(a.rjust(18) for a in arms))
        print("  " + "ALL".ljust(12) + "".join(str(s[a]).rjust(18) for a in arms))
        for g, gs in s["by_group"].items():
            print(
                "  "
                + f"{g} (n={gs['n']})".ljust(12)
                + "".join(str(gs[a]).rjust(18) for a in arms)
            )


# ── Estimate (no LLM) ────────────────────────────────────────────────


def run_estimate(args: argparse.Namespace, cases: list[e1.Case]) -> None:
    sys_chars = _system_message_chars()
    calls = 0
    tokens = 0
    for case in cases:
        text = _read_text(case)
        if text is None:
            continue
        for variant in args.variants:
            win = build_input(text, variant, args.budget_tokens)
            per_call = (
                sys_chars + len(win.text)
            ) // CHARS_PER_TOKEN + CLASSIFY_OUTPUT_TOKENS_EST
            calls += args.reps
            tokens += per_call * args.reps
    worst = 1 + settings.classify_max_retries
    print(f"cases={len(cases)} variants={args.variants} reps={args.reps}")
    print(
        f"first-attempt calls: {calls}  (worst case with every retry used: {calls * worst})"
    )
    print(
        f"estimated tokens (≈ chars/{CHARS_PER_TOKEN}): {tokens:,}  (worst case: {tokens * worst:,})"
    )
    print(
        "Multiply by the provider's current per-token price; check it before running."
    )


# ── Real run ─────────────────────────────────────────────────────────


def _record(
    case: e1.Case, variant: str, rep: int, state: dict[str, Any]
) -> dict[str, Any]:
    exp = e1.expected_for(case)
    pred = e1.predicted_from_state(state)
    steps = state.get("step_runs") or []
    classify_steps = [s for s in steps if s["stage"] == "classify"]
    checks = [s for s in steps if s["stage"] == "classify_validate"]
    first_check = checks[0] if checks else None
    first_label = (
        (classify_steps[0].get("output") or {}).get("label") if classify_steps else None
    )
    first_quote_pass: Optional[bool] = None
    if first_check is not None and first_label in ("M&A", "Dividend"):
        first_quote_pass = first_check["outcome"] == "passed"
    tokens = [
        s.get("total_tokens")
        for s in classify_steps
        if s.get("total_tokens") is not None
    ]
    latency = [
        s.get("latency_ms") for s in classify_steps if s.get("latency_ms") is not None
    ]
    window = state.get("classifier_input") or ""
    return {
        "case_id": case.case_id,
        "group": case.group,
        "variant": variant,
        "rep": rep,
        "verified": case.verified,
        "ambiguous": case.ambiguous,
        "gold_projected": list(e1.project(case.labels)),
        "expected": exp.describe() + (" | unclear" if exp.accept_unclear else ""),
        "multi": exp.multi,
        "predicted": pred,
        "correct": e1.is_correct(pred, exp),
        "coverage": e1.coverage_for(case, window) if case.verified else None,
        "classify_attempts": len(classify_steps),
        "first_quote_pass": first_quote_pass,
        "tokens": sum(tokens) if tokens else None,
        "latency_ms": sum(latency) if latency else None,
        "run_id": state.get("run_id"),
        "error_code": (state.get("failure") or {}).get("code"),
    }


async def run_real(args: argparse.Namespace, cases: list[e1.Case]) -> Path:
    from src.graph.builder import build_classification_graph
    from src.nodes.reader import read_single_file
    from src.storage.db import init_db
    from src.storage.repository import ExtractionRepository

    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY is not set")
    out = _out_dir(args.out, args.tag)
    settings.sqlite_db_path = str(out / "run.db")
    settings.raw_store_path = str(out / "raw")
    settings.chromadb_path = str(out / "chroma")
    settings.rag_enabled = False
    settings.classify_temperature = args.temperature
    settings.classifier_input_tokens = args.budget_tokens
    await init_db()
    repo = ExtractionRepository()
    graph = build_classification_graph()
    (out / "manifest.json").write_text(
        json.dumps(_manifest(args, cases, "run"), indent=2), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    calls_used = 0
    stopped_early = False
    sem = asyncio.Semaphore(args.concurrency)

    async def one(case: e1.Case, variant: str, rep: int) -> Optional[dict]:
        async with sem:
            state = await read_single_file(str(case.input_file))
            final = await graph.ainvoke(state)
            rec = _record(case, variant, rep, final)
            await repo.save_steps_only(
                final, status=f"e1:{variant}:rep{rep}:{rec['predicted']}"
            )
            return rec

    started = time.perf_counter()
    for variant in args.variants:
        settings.classifier_input_strategy = variant
        for rep in range(1, args.reps + 1):
            worst = len(cases) * (1 + settings.classify_max_retries)
            if calls_used + worst > args.max_calls:
                stopped_early = True
                print(
                    f"STOP: {variant} rep {rep} could exceed --max-calls={args.max_calls}"
                )
                break
            batch = await asyncio.gather(*(one(c, variant, rep) for c in cases))
            for rec in batch:
                records.append(rec)
                calls_used += rec["classify_attempts"]
            print(f"  {variant} rep {rep}: done ({calls_used} calls so far)")
        if stopped_early:
            break

    _write_csv(out / "per_case.csv", [_redact_record(r) for r in records])
    summary: dict[str, Any] = {
        "calls_used": calls_used,
        "stopped_early": stopped_early,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "variants": {},
        "paired": {},
    }
    by_variant = {v: [r for r in records if r["variant"] == v] for v in args.variants}
    for v, rows in by_variant.items():
        if rows:
            summary["variants"][v] = {
                **e1.summarize(rows),
                "rep_agreement": e1.rep_agreement(rows),
            }
    names = [v for v in args.variants if by_variant[v]]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            summary["paired"][f"{a} vs {b}"] = e1.mcnemar(by_variant[a], by_variant[b])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ("calls_used", "stopped_early", "paired")}, indent=2
        )
    )
    print(f"\nWritten to {out}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 classification evaluation")
    p.add_argument(
        "--dataset", type=Path, required=True, help="E1 JSONL (e1_dataset_draft.jsonl)"
    )
    p.add_argument(
        "--inputs-dir", type=Path, default=None, help="default: <dataset dir>/inputs"
    )
    p.add_argument("--split", default="dev")
    p.add_argument(
        "--case-ids", default=None, help="comma-separated ids to freeze the case list"
    )
    p.add_argument(
        "--max-case-number",
        type=int,
        default=None,
        help="only E1-001..E1-<N> (e.g. 60 = the original set)",
    )
    p.add_argument("--variants", default="head,structured,keyword")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--budget-tokens", type=int, default=3500)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-calls", type=int, default=400)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "e1")
    p.add_argument("--tag", default="dev")
    p.add_argument("--coverage-only", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--allow-test", action="store_true")
    args = p.parse_args(argv)
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in args.variants if v not in STRATEGIES]
    if unknown:
        p.error(f"unknown variants {unknown}; choose from {STRATEGIES}")
    if args.split == "test" and not args.allow_test:
        p.error(
            "the test split is sealed until the prompt and input strategy are final (--allow-test)"
        )
    args.inputs_dir = args.inputs_dir or args.dataset.parent / "inputs"
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    ids = args.case_ids.split(",") if args.case_ids else None
    cases = e1.load_cases(args.dataset, args.inputs_dir, args.split, ids)
    if args.max_case_number is not None:
        cases = [
            c for c in cases if int(c.case_id.split("-")[1]) <= args.max_case_number
        ]
    if not cases:
        print("no cases selected")
        return 1
    print(
        f"{len(cases)} {args.split} cases ({sum(c.verified for c in cases)} verified)"
    )
    if args.coverage_only:
        run_coverage(args, cases)
    elif args.estimate:
        run_estimate(args, cases)
    else:
        asyncio.run(run_real(args, cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
