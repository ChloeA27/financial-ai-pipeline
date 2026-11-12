"""
Data shaping for the E4 dashboard (``src/e4_dashboard.py``).

Plain Python, no Streamlit and no pandas, so it is unit-tested in CI.

Reads one E4 run directory (``results/e4/<timestamp>_<tag>/``):
``summary.json`` for the funnel and reasons, and the per-document table from
``_local/per_case_full.csv`` when present (local only, has every label and
answer). It falls back to the committed ``per_case.csv``, where unverified
cases carry no labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

OUTCOME_ORDER = ("stored", "human_review", "out_of_scope", "dlq")
OUTCOME_LABELS = {
    "stored": "Stored",
    "human_review": "Human review",
    "out_of_scope": "Out of scope",
    "dlq": "DLQ (engineering)",
}
GOLD_ORDER = ("mna", "dividend", "dividend+mna", "none")


def list_runs(results_dir: Path) -> list[Path]:
    """Run directories that have a summary, newest first."""
    if not Path(results_dir).is_dir():
        return []
    runs = [p for p in Path(results_dir).iterdir() if (p / "summary.json").is_file()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_cases(run_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """(rows, full): ``full`` is True when labels of unverified cases are present."""
    local = Path(run_dir) / "_local" / "per_case_full.csv"
    path = local if local.is_file() else Path(run_dir) / "per_case.csv"
    if not path.is_file():
        return [], False
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["verified"] = r.get("verified") == "True"
    return rows, path == local


def kpis(scope: dict[str, Any]) -> dict[str, Optional[float]]:
    """Headline numbers of one label scope."""
    n = scope["n"]
    out = {"documents": n, "classification_accuracy": scope["classification_accuracy"]}
    for o in OUTCOME_ORDER:
        out[o] = scope["outcomes"].get(o, 0)
    out["supported_stored_rate"] = scope.get("supported_stored_rate")
    return out


def funnel_rows(scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Long rows (gold, outcome, count, share) for a stacked bar per gold label."""
    by_gold = scope["outcomes_by_gold"]
    golds = [g for g in GOLD_ORDER if g in by_gold] + sorted(
        g for g in by_gold if g not in GOLD_ORDER
    )
    rows = []
    for g in golds:
        total = sum(by_gold[g].values())
        for o in OUTCOME_ORDER:
            c = by_gold[g].get(o, 0)
            rows.append(
                {
                    "gold": g,
                    "outcome": OUTCOME_LABELS[o],
                    "count": c,
                    "share": round(c / total, 4) if total else 0.0,
                    "total": total,
                }
            )
    return rows


def reason_rows(scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Why documents went to human review or the DLQ, most frequent first."""
    return [
        {
            "outcome": OUTCOME_LABELS.get(r["outcome"], r["outcome"]),
            "stage": r["stage"],
            "code": r["code"],
            "count": r["count"],
        }
        for r in scope["reasons"]
    ]
