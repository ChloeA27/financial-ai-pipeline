"""
Data shaping for the E1 dashboard (``src/e1_dashboard.py``).

Plain Python — no Streamlit, no pandas — so it is unit-tested in CI, where the
optional dashboard dependencies are not installed.

Three levels of view, all read from one E1 run directory
(``results/e1/<timestamp>_<tag>/``):

  * meso  — variant comparison: ``summary.json``
  * macro — outcome funnel / retry causes / token use: ``run.db`` step_runs
  * micro — one document's trace: its step_runs plus the classifier window,
            rebuilt deterministically from the stored raw bytes

Blind-labelling protection: case-level views use ``per_case.csv``, where cases
that are not human-verified carry no labels or answers; the trace view lists
them only when explicitly asked to.
"""

from __future__ import annotations

import csv
import html
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from src.nodes.input_window import build_input
from src.nodes.reader import SourceDecodeError, decode_source

VARIANT_ORDER = ("head", "structured", "keyword")
VARIANT_LABELS = {
    "head": "V0 head",
    "structured": "V1 structured",
    "keyword": "V2 keyword",
}

# (key in summary, label, kind) — kind drives formatting: "rate", "count", "tokens", "ms"
METRICS: list[tuple[str, str, str]] = [
    ("accuracy_verified", "Accuracy (verified cases)", "rate"),
    ("n_scored_verified", "Scored runs (verified × reps)", "count"),
    ("none_specificity", "none specificity", "rate"),
    ("input_coverage_verified", "Input coverage (verified positives)", "rate"),
    ("first_attempt_quote_pass_rate", "Quote accepted on 1st attempt", "rate"),
    ("classify_retry_rate", "Docs needing a classify retry", "rate"),
    ("classify_failed_rate", "Docs failing classification", "rate"),
    ("unclear_rate_non_ambiguous", "unclear rate (non-ambiguous)", "rate"),
    ("rep_agreement", "Agreement between repetitions", "rate"),
    ("tokens_per_doc_mean", "Tokens per document", "tokens"),
    ("latency_ms_per_doc_mean", "Latency per document", "ms"),
]

OUTCOME_ORDER = (
    "accepted · 1st attempt",
    "accepted · after retry",
    "ended · none",
    "ended · unclear",
    "failed",
)


# ── Run discovery and loading ────────────────────────────────────────


def list_runs(base: Path) -> list[Path]:
    """Real (LLM) E1 runs under ``base``, newest first."""
    runs = []
    for d in Path(base).glob("*/"):
        manifest = d / "manifest.json"
        if (d / "summary.json").exists() and manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")).get("mode") == "run":
                runs.append(d)
    return sorted(runs, reverse=True)


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    per_case_path = run_dir / "per_case.csv"
    per_case = []
    if per_case_path.exists():
        with open(per_case_path, encoding="utf-8") as fh:
            per_case = list(csv.DictReader(fh))
    db = run_dir / "run.db"
    return {
        "dir": run_dir,
        "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
        "per_case": per_case,
        "db": db if db.exists() else None,
    }


def variants_in(summary: dict[str, Any]) -> list[str]:
    present = summary.get("variants", {})
    return [v for v in VARIANT_ORDER if v in present] + [
        v for v in present if v not in VARIANT_ORDER
    ]


# ── Meso: variant comparison ─────────────────────────────────────────


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "rate":
        return f"{value * 100:.1f}%"
    if kind == "tokens":
        return f"{value:,.0f}"
    if kind == "ms":
        return f"{value / 1000:.1f} s"
    return str(value)


def metric_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per metric, one column per variant (formatted strings)."""
    rows = []
    for key, label, kind in METRICS:
        row: dict[str, Any] = {"Metric": label}
        for v in variants_in(summary):
            row[VARIANT_LABELS.get(v, v)] = fmt(summary["variants"][v].get(key), kind)
        rows.append(row)
    return rows


def rate_series(summary: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    """Long-format rows (metric, variant, value) for a grouped bar chart."""
    labels = {k: lab for k, lab, _ in METRICS}
    out = []
    for key in keys:
        for v in variants_in(summary):
            value = summary["variants"][v].get(key)
            if value is not None:
                out.append({"metric": labels[key], "variant": v, "value": value})
    return out


def group_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    groups: set[str] = set()
    for v in variants_in(summary):
        groups |= set(summary["variants"][v].get("by_group", {}))
    rows = []
    for g in sorted(groups):
        row: dict[str, Any] = {"Group": g}
        for v in variants_in(summary):
            cell = summary["variants"][v].get("by_group", {}).get(g, {})
            n = cell.get("n_scored", 0)
            row[VARIANT_LABELS.get(v, v)] = (
                f"{fmt(cell.get('accuracy'), 'rate')} (n={n})"
                if n
                else "not labelled yet"
            )
        rows.append(row)
    return rows


def paired_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair, m in summary.get("paired", {}).items():
        a, _, b = pair.partition(" vs ")
        rows.append(
            {
                "Comparison": f"{VARIANT_LABELS.get(a, a)} vs {VARIANT_LABELS.get(b, b)}",
                "Paired cases": m["n_paired"],
                "A right, B wrong": m["a_right_b_wrong"],
                "A wrong, B right": m["a_wrong_b_right"],
                "p (exact McNemar)": m["p_value"],
            }
        )
    return rows


# ── Macro: outcomes from step_runs ───────────────────────────────────


def step_rows(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM step_runs ORDER BY id")]
    finally:
        conn.close()
    for r in rows:
        r["output"] = json.loads(r["output"]) if r.get("output") else None
    return rows


def run_index(per_case: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """run_id → its per_case row (case, variant, rep, verified, …)."""
    return {r["run_id"]: r for r in per_case if r.get("run_id")}


def outcome_of_run(steps: list[dict[str, Any]]) -> str:
    """Where one classification run ended, from its own step records."""
    checks = [s for s in steps if s["stage"] == "classify_validate"]
    last = checks[-1] if checks else (steps[-1] if steps else None)
    if last is None:
        return "failed"
    if last["stage"] == "classify_validate" and last["outcome"] == "passed":
        return (
            "accepted · 1st attempt"
            if last["attempt"] == 1
            else "accepted · after retry"
        )
    if last.get("error_code") == "UNSUPPORTED_DOC_TYPE":
        return "ended · none"
    if last.get("error_code") == "CLASSIFICATION_UNCLEAR":
        return "ended · unclear"
    return "failed"


def outcomes_by_variant(
    rows: list[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> dict[str, Counter]:
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    out: dict[str, Counter] = defaultdict(Counter)
    for run_id, steps in by_run.items():
        meta = index.get(run_id)
        if meta:
            out[meta["variant"]][outcome_of_run(steps)] += 1
    return out


def retry_causes(
    rows: list[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> dict[str, Counter]:
    """Why attempts were rejected (retryable) — terminal labels excluded."""
    terminal = {"UNSUPPORTED_DOC_TYPE", "CLASSIFICATION_UNCLEAR"}
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        meta = index.get(r["run_id"])
        if meta and r["outcome"] != "passed" and r.get("error_code") not in terminal:
            out[meta["variant"]][r.get("error_code") or "UNKNOWN"] += 1
    return out


def llm_calls(
    rows: list[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per classifier call: variant, completion tokens, outcome."""
    out = []
    for r in rows:
        meta = index.get(r["run_id"])
        if meta and r["stage"] == "classify" and r.get("completion_tokens") is not None:
            out.append(
                {
                    "variant": meta["variant"],
                    "completion_tokens": r["completion_tokens"],
                    "prompt_tokens": r.get("prompt_tokens"),
                    "parsed": r["outcome"] == "passed",
                }
            )
    return out


# ── Micro: one document ──────────────────────────────────────────────


def selectable_cases(
    per_case: list[dict[str, Any]], include_unverified: bool
) -> list[dict[str, Any]]:
    rows = [r for r in per_case if include_unverified or r.get("verified") == "True"]
    return sorted(rows, key=lambda r: (r["case_id"], r["variant"], int(r["rep"])))


def trace(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["run_id"] == run_id]


def window_for(
    run_dir: Path, steps: list[dict[str, Any]], budget_tokens: int
) -> Optional[dict[str, Any]]:
    """Rebuild the classifier window from the stored raw bytes (deterministic)."""
    first = next((s for s in steps if s["stage"] == "classify"), None)
    if first is None:
        return None
    stats = (first.get("output") or {}).get("input") or {}
    strategy = stats.get("strategy", "head")
    raw = Path(run_dir) / "raw" / f"{first['source_hash']}.txt"
    source = raw if raw.exists() else Path(first.get("file_path") or "")
    if not source.exists():
        return None
    try:
        text = decode_source(source.read_bytes())[0]
    except SourceDecodeError:
        return None
    window = build_input(text, strategy, budget_tokens)
    return {"text": window.text, "stats": stats or window.stats()}


_TYPO = {"'": "['’‘]", '"': '["“”]', "-": "[-–—]"}


def _quote_pattern(quote: str) -> Optional[re.Pattern]:
    words = re.findall(r"\S+", quote.strip(" .…\"'"))
    if not words:
        return None
    parts = []
    for w in words:
        parts.append("".join(_TYPO.get(ch, re.escape(ch)) for ch in w))
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


def highlight(text: str, quotes: list[tuple[str, str]]) -> tuple[str, list[bool]]:
    """
    HTML-escape ``text`` and wrap each quote's first match in
    ``<mark class="<css>">``.  Returns (html, found flags in input order).
    Overlapping matches keep the earlier quote.
    """
    spans: list[tuple[int, int, str]] = []
    found = []
    for quote, css in quotes:
        pat = _quote_pattern(quote or "")
        m = pat.search(text) if pat else None
        found.append(m is not None)
        if m and all(m.end() <= s or m.start() >= e for s, e, _ in spans):
            spans.append((m.start(), m.end(), css))
    spans.sort()
    out, pos = [], 0
    for start, end, css in spans:
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark class="{css}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out), found
