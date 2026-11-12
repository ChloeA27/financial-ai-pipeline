"""
E4 runner: the whole pipeline (read → classify → extract → validate → route)
on the E1 dev filings, scored with ``evals/evaluators/e4.py``.

    python -m evals.run_e4 --estimate            # no LLM: case count and token estimate
    python -m evals.run_e4 --tag silver_dev      # real run on the dev split

Dataset: the E1 JSONL (``--dataset`` or ``E1_DATASET``); it lives outside the repo
and is only read. Only the dev split can be run: the test split stays untouched
until the prompts are final.

Writes ``results/e4/<UTC timestamp>_<tag>/``:
  manifest.json      model, settings, dataset SHA-256, git commit
  summary.json       per label scope: classification accuracy, outcome funnel, reasons
  per_case.csv       committed; cases that are not human-verified carry no labels
                     or answers (blind-labelling rule, as in results/e1)
  _local/            NOT committed: per_case_full.csv with every label and answer,
                     for the dashboard (``make dash``)
  run.db, raw/       NOT committed: the pipeline database and source bytes
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.evaluators import e1, e4  # noqa: E402
from evals.run_e1 import _default_dataset, _git_state, _sha256_file  # noqa: E402
from src.config import settings  # noqa: E402

RESULTS = REPO_ROOT / "results" / "e4"
SPLIT = "dev"

# Token estimate per call, from measured runs (E1 run 2 head: 1,682 per classified
# document; scenario run 2: about 1,000-1,400 per extraction on short texts).
# The extractor sees at most 6,000 characters, so long filings cost about the same.
EST_CLASSIFY = 1_700
EST_EXTRACT = 2_500
WORST_CALLS_PER_DOC = 3 + 4  # classify 1 + 2 retries, extract 1 + 3 retries

REDACTED_KEYS = (
    "case_id",
    "verified",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
)


def _estimate(cases: list[e1.Case]) -> dict[str, Any]:
    positives = sum(1 for c in cases if e4.gold_label(c) in ("mna", "dividend"))
    expected = len(cases) * EST_CLASSIFY + positives * EST_EXTRACT
    return {
        "cases": len(cases),
        "verified": sum(c.verified for c in cases),
        "labelled_supported": positives,
        "expected_tokens": expected,
        "worst_calls": len(cases) * WORST_CALLS_PER_DOC,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


async def _run(args: argparse.Namespace, cases: list[e1.Case]) -> Path:
    from src.storage.db import init_db
    from src.storage.repository import ExtractionRepository
    from src.workflows.graph import run_pipeline

    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY is not set")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{stamp}_{args.tag}"
    (out / "_local").mkdir(parents=True)
    settings.sqlite_db_path = str(out / "run.db")
    settings.raw_store_path = str(out / "raw")
    settings.chromadb_path = str(out / "chroma")
    settings.rag_enabled = False
    if args.model:
        settings.deepseek_model = args.model
    await init_db()
    repo = ExtractionRepository()

    manifest = {
        "created_at_utc": stamp,
        "mode": "run",
        "dataset_file": args.dataset.name,
        "dataset_sha256": _sha256_file(args.dataset),
        "split": SPLIT,
        "cases": len(cases),
        "case_ids": [c.case_id for c in cases],
        "model": settings.deepseek_model,
        "classifier_input_strategy": settings.classifier_input_strategy,
        "classify_thinking": settings.classify_thinking,
        "classify_max_retries": settings.classify_max_retries,
        "max_retries": settings.max_retries,
        "rag_enabled": settings.rag_enabled,
        "estimate": _estimate(cases),
        "git": _git_state(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(case: e1.Case) -> dict[str, Any]:
        nonlocal done
        async with sem:
            state = await run_pipeline(str(case.input_file))
            await repo.save_extraction(state)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(cases)} documents")
            return e4.record(case, state)

    started = time.perf_counter()
    records = await asyncio.gather(*(one(c) for c in cases))
    summary = {
        "wall_seconds": round(time.perf_counter() - started, 1),
        "llm_tokens": {
            "prompt": sum(r["prompt_tokens"] for r in records),
            "completion": sum(r["completion_tokens"] for r in records),
        },
        "scopes": e4.summarize(records),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(out / "_local" / "per_case_full.csv", records)
    _write_csv(
        out / "per_case.csv",
        [r if r["verified"] else {k: r[k] for k in REDACTED_KEYS} for r in records],
    )
    return out


def _print(summary: dict[str, Any]) -> None:
    for scope, s in summary["scopes"].items():
        acc = s["classification_accuracy"]
        print(
            f"[{scope}] n={s['n']}  classification "
            f"{'-' if acc is None else f'{acc:.1%}'} of {s['classification_scored']}  "
            f"outcomes {s['outcomes']}"
        )
    print(f"tokens: {summary['llm_tokens']}  wall: {summary['wall_seconds']} s")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4: whole pipeline on the E1 dev split")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--inputs-dir", type=Path, default=None)
    p.add_argument("--tag", default="dev")
    p.add_argument("--model", default=None, help="override DEEPSEEK_MODEL")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="first N cases only")
    p.add_argument("--max-calls", type=int, default=800)
    p.add_argument("--estimate", action="store_true", help="no LLM: print the estimate")
    args = p.parse_args(argv)
    args.dataset = args.dataset or _default_dataset()
    if args.dataset is None:
        p.error("no dataset: pass --dataset or set E1_DATASET")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    inputs = args.inputs_dir or args.dataset.parent / "inputs"
    cases = e1.load_cases(args.dataset, inputs, SPLIT)[: args.limit]
    est = _estimate(cases)
    print(json.dumps(est, indent=2))
    if args.estimate:
        return 0
    if est["worst_calls"] > args.max_calls:
        print(
            f"STOP: worst case {est['worst_calls']} calls > --max-calls {args.max_calls}"
        )
        return 1
    out = asyncio.run(_run(args, cases))
    _print(json.loads((out / "summary.json").read_text()))
    print(f"written: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
