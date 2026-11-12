"""
Field presence in real filings: how often does each candidate extraction field
appear in the text at all?  Keyword search only (no LLM, no tokens).

Used to decide a field's presence level (REQUIRED / EXPECTED / OPTIONAL).
A keyword hit is a *mention*, not proof that the filing discloses the value:
read the counts as upper bounds and spot-check before quoting them.

Only the dev split is read, on purpose: design decisions must not look at the
test split.  Labels come from the E1 dataset; most are unverified drafts, so
every count is reported twice: human-verified cases only, and all cases.

    python -m evals.analysis.field_presence            # dataset from E1_DATASET
    python -m evals.analysis.field_presence --dataset <dir>/e1_dataset_draft.jsonl

Writes ``results/field_presence/<UTC timestamp>_dev/``: ``manifest.json``,
``summary.json`` and ``per_case.csv`` (hit flags only, no filing text).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evals.evaluators.e1 import Case, load_cases
from evals.run_e1 import _default_dataset, _git_state, _sha256_file
from src.nodes.reader import decode_source

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "results" / "field_presence"
SPLIT = "dev"

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
)
_DATE = rf"{_MONTH}\.? \d{{1,2}},? \d{{4}}"

# label -> field -> case-insensitive pattern
PATTERNS: dict[str, dict[str, str]] = {
    "dividend": {
        "ex_dividend_date": r"ex-?\s?dividend|ex-?date",
        "record_date": r"record date|of record|holders of record",
        "payment_date": r"payable|payment date|paid on|to be paid",
        "per_share_amount": (
            r"\$\s?\d+(?:\.\d+)?\s*(?:per|a)\s+(?:common\s+|ordinary\s+)?share"
        ),
        "preferred_mentioned": r"preferred (?:stock|shares)|series [A-Z] ",
        "trading_symbol_on_cover": r"trading symbol",
    },
    "mna": {
        "dollar_amount_million_billion": r"\$\s?\d[\d,.]*\s*(?:million|billion)",
        "consideration_wording": (
            r"cash consideration|merger consideration|per share in cash|"
            r"exchange ratio|shares of .{0,40}common stock for each"
        ),
        "agreement_dated": (
            r"(?:agreement and plan of merger|purchase agreement|merger agreement)"
            rf"[^.]{{0,80}}dated (?:as of )?{_DATE}"
        ),
        "exhibit_2_1": r"exhibit 2\.1|\b2\.1\b[^.]{0,60}agreement",
        "termination_fee": r"termination fee",
        "trading_symbol_on_cover": r"trading symbol",
    },
}


def _text(case: Case) -> str:
    return re.sub(r"\s+", " ", decode_source(case.input_file.read_bytes())[0])


def scan(cases: list[Case]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-case hit flags and per-label counts (verified-only and all)."""
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for label, patterns in PATTERNS.items():
        compiled = {f: re.compile(p, re.I) for f, p in patterns.items()}
        counts = {"verified": {"n": 0}, "all": {"n": 0}}
        for case in cases:
            if label not in case.labels:
                continue
            text = _text(case)
            hits = {f: bool(rx.search(text)) for f, rx in compiled.items()}
            rows.append(
                {"case_id": case.case_id, "label": label, "verified": case.verified}
                | hits
            )
            for scope in ("verified", "all"):
                if scope == "verified" and not case.verified:
                    continue
                c = counts[scope]
                c["n"] += 1
                for f, hit in hits.items():
                    c[f] = c.get(f, 0) + int(hit)
        summary[label] = {
            scope: {
                "n": c["n"],
                "fields": {
                    f: {
                        "hits": c.get(f, 0),
                        "rate": round(c.get(f, 0) / c["n"], 4) if c["n"] else None,
                    }
                    for f in patterns
                },
            }
            for scope, c in counts.items()
        }
    return rows, summary


def main(argv: Optional[list[str]] = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--inputs-dir", type=Path, default=None)
    args = p.parse_args(argv)

    dataset = args.dataset or _default_dataset()
    if dataset is None:
        p.error("no dataset: pass --dataset or set E1_DATASET")
    inputs_dir = args.inputs_dir or dataset.parent / "inputs"
    cases = load_cases(dataset, inputs_dir, SPLIT)

    rows, summary = scan(cases)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = OUT_ROOT / f"{stamp}_{SPLIT}"
    out.mkdir(parents=True)
    manifest = {
        "created_at_utc": stamp,
        "dataset_file": dataset.name,
        "dataset_sha256": _sha256_file(dataset),
        "split": SPLIT,
        "cases_in_split": len(cases),
        "method": "case-insensitive regex over whitespace-normalised text; no LLM",
        "patterns": PATTERNS,
        "git": _git_state(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r} - {"case_id", "label", "verified"})
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "label", "verified", *fields])
        w.writeheader()
        w.writerows(rows)

    for label, scopes in summary.items():
        for scope, s in scopes.items():
            print(f"[{label}] {scope} n={s['n']}")
            for f, v in s["fields"].items():
                rate = f"{v['rate']:.0%}" if v["rate"] is not None else "-"
                print(f"  {f:32s} {v['hits']:3d}/{s['n']}  {rate}")
    print(f"\nwritten: {out.relative_to(REPO_ROOT)}")
    return out


if __name__ == "__main__":
    main()
