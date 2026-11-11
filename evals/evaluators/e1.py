"""
E1 — classification-stage evaluation: scoring rules and metrics (pure code).

Gold comes from the E1 dataset (SEC EDGAR filings, labelling guide v0.3+),
whose label set is richer than this pipeline's:

    gold:      mna, dividend, mgmt_change, none   (multi-label)
    pipeline:  M&A, Dividend, none, unclear        (single label)

**Projection rule** (decided 2026-11-10): gold labels are projected onto the
two supported types — ``gold ∩ {mna, dividend}``, and an empty result means
``none``.  So a ``mgmt_change``-only filing is expected to come back ``none``,
and ``mna + mgmt_change`` is expected to come back ``mna``.  When both
supported labels remain (``mna + dividend``), a single-label classifier cannot
match exactly; ``unclear`` is accepted and such cases are reported separately
as ``multi``.

Only rows whose gold is human-verified count towards accuracy; the others
still contribute label-free metrics (unclear rate, failures, tokens, ...).
API / input errors are counted separately and never as wrong answers.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from src.nodes.classification_validator import normalise_text
from src.nodes.input_window import evidence_in_window

SUPPORTED = ("mna", "dividend")
PIPELINE_TO_E1 = {
    "M&A": "mna",
    "Dividend": "dividend",
    "none": "none",
    "unclear": "unclear",
}
PROJECTION_RULE = (
    "gold ∩ {mna, dividend}; empty → none (mgmt_change-only filings count as none); "
    "mna+dividend accepts 'unclear' (single-label classifier)"
)
ERROR_CATEGORIES = {"transient", "input_error"}


# ── Dataset ──────────────────────────────────────────────────────────


@dataclass
class Case:
    case_id: str
    split: str
    group: str
    adsh: str
    input_file: Path
    verified: bool
    labels: list[str]
    ambiguous: bool = False
    acceptable_labels: list[list[str]] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)
    guide_version: Optional[str] = None


def load_cases(
    dataset_path: Path,
    inputs_dir: Path,
    split: str,
    case_ids: Optional[Iterable[str]] = None,
) -> list[Case]:
    """Read the E1 JSONL; keep one split (and optionally a fixed list of ids)."""
    wanted = set(case_ids) if case_ids else None
    cases = []
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        if wanted is not None and row["case_id"] not in wanted:
            continue
        cases.append(
            Case(
                case_id=row["case_id"],
                split=row["split"],
                group=row.get("group", ""),
                adsh=row["adsh"],
                input_file=Path(inputs_dir) / f"{row['adsh']}.txt",
                verified=row.get("status") == "verified",
                labels=list(row.get("labels") or []),
                ambiguous=bool(row.get("ambiguous")),
                acceptable_labels=[list(a) for a in row.get("acceptable_labels") or []],
                evidence=list(row.get("evidence") or []),
                guide_version=row.get("guide_version"),
            )
        )
    return sorted(cases, key=lambda c: c.case_id)


# ── Scoring ──────────────────────────────────────────────────────────


def project(labels: Iterable[str]) -> tuple[str, ...]:
    """Project gold labels onto the supported types (see module docstring)."""
    kept = tuple(sorted({label for label in labels if label in SUPPORTED}))
    return kept or ("none",)


@dataclass
class Expected:
    answers: set[tuple[str, ...]]
    accept_unclear: bool
    multi: bool

    def describe(self) -> str:
        return " | ".join("+".join(a) for a in sorted(self.answers))


def expected_for(case: Case) -> Expected:
    primary = project(case.labels)
    answers = {primary}
    if case.ambiguous:
        answers |= {project(a) for a in case.acceptable_labels}
    multi = len(primary) > 1
    return Expected(
        answers=answers, accept_unclear=case.ambiguous or multi, multi=multi
    )


def predicted_from_state(state: dict[str, Any]) -> str:
    """
    The classification outcome in E1 vocabulary:
    ``mna`` / ``dividend`` / ``none`` / ``unclear``, ``failed:<CODE>`` when the
    classify loop gave up, ``error`` for API / input problems.
    """
    failure = state.get("failure") or {}
    if failure.get("category") in ERROR_CATEGORIES:
        return "error"
    doc_type = state.get("doc_type")
    if state.get("classification_passed") is True and doc_type in PIPELINE_TO_E1:
        return PIPELINE_TO_E1[doc_type]
    return f"failed:{failure.get('code') or 'UNKNOWN'}"


def is_correct(predicted: str, expected: Expected) -> Optional[bool]:
    """None = not scoreable (infrastructure error)."""
    if predicted == "error":
        return None
    if predicted == "unclear":
        return expected.accept_unclear
    if predicted.startswith("failed:"):
        return False
    return (predicted,) in expected.answers


def coverage_for(case: Case, window_text: str) -> Optional[bool]:
    """Input coverage: is the gold evidence for the supported labels in the window?"""
    quotes = [e["quote"] for e in case.evidence if e.get("label") in SUPPORTED]
    return evidence_in_window(quotes, window_text, normalise_text)


# ── Metrics ──────────────────────────────────────────────────────────


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for ONE variant (all reps pooled unless filtered beforehand)."""
    scored = [r for r in records if r["verified"] and r["correct"] is not None]
    errors = [r for r in records if r["predicted"] == "error"]

    per_label: dict[str, dict[str, Optional[float]]] = {}
    for label in SUPPORTED:
        tp = sum(
            1
            for r in scored
            if r["predicted"] == label and label in r["gold_projected"]
        )
        fp = sum(
            1
            for r in scored
            if r["predicted"] == label and label not in r["gold_projected"]
        )
        fn = sum(
            1
            for r in scored
            if label in r["gold_projected"] and r["predicted"] != label
        )
        per_label[label] = {
            "precision": _rate(tp, tp + fp),
            "recall": _rate(tp, tp + fn),
            "support": tp + fn,
        }

    negatives = [
        r for r in scored if r["gold_projected"] == ["none"] and not r["ambiguous"]
    ]
    by_group: dict[str, dict[str, Any]] = {}
    for group in sorted({r["group"] for r in records}):
        rows = [r for r in scored if r["group"] == group]
        by_group[group] = {
            "n_scored": len(rows),
            "accuracy": _rate(sum(r["correct"] for r in rows), len(rows)),
        }

    covered = [r for r in records if r["verified"] and r["coverage"] is not None]
    supported_first = [r for r in records if r["first_quote_pass"] is not None]
    non_amb = [r for r in records if not r["ambiguous"] and r["predicted"] != "error"]
    confusion = Counter(("+".join(r["gold_projected"]), r["predicted"]) for r in scored)
    tokens = [r["tokens"] for r in records if r["tokens"] is not None]
    latency = [r["latency_ms"] for r in records if r["latency_ms"] is not None]

    return {
        "n_runs": len(records),
        "n_scored_verified": len(scored),
        "accuracy_verified": _rate(sum(r["correct"] for r in scored), len(scored)),
        "per_label": per_label,
        "none_specificity": _rate(
            sum(1 for r in negatives if r["predicted"] == "none"), len(negatives)
        ),
        "by_group": by_group,
        "unclear_rate_non_ambiguous": _rate(
            sum(1 for r in non_amb if r["predicted"] == "unclear"), len(non_amb)
        ),
        "classify_failed_rate": _rate(
            sum(1 for r in records if r["predicted"].startswith("failed:")),
            len(records),
        ),
        "failed_by_code": dict(
            Counter(
                r["predicted"][7:]
                for r in records
                if r["predicted"].startswith("failed:")
            )
        ),
        "first_attempt_quote_pass_rate": _rate(
            sum(r["first_quote_pass"] for r in supported_first), len(supported_first)
        ),
        "classify_retry_rate": _rate(
            sum(1 for r in records if r["classify_attempts"] > 1), len(records)
        ),
        "input_coverage_verified": _rate(
            sum(r["coverage"] for r in covered), len(covered)
        ),
        "input_coverage_n": len(covered),
        "errors": len(errors),
        "tokens_per_doc_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "tokens_total": sum(tokens) if tokens else None,
        "latency_ms_per_doc_mean": (
            round(sum(latency) / len(latency)) if latency else None
        ),
        "confusion_verified": {
            f"{g} -> {p}": n for (g, p), n in sorted(confusion.items())
        },
    }


def rep_agreement(records: list[dict[str, Any]]) -> Optional[float]:
    """Share of cases whose prediction is identical in every repetition."""
    by_case: dict[str, set[str]] = defaultdict(set)
    reps: dict[str, int] = Counter()
    for r in records:
        by_case[r["case_id"]].add(r["predicted"])
        reps[r["case_id"]] += 1
    multi = [c for c in by_case if reps[c] > 1]
    if not multi:
        return None
    return _rate(sum(1 for c in multi if len(by_case[c]) == 1), len(multi))


def mcnemar(records_a: list[dict], records_b: list[dict]) -> dict[str, Any]:
    """
    Paired comparison on verified cases (first repetition of each variant).

    b = A right & B wrong, c = A wrong & B right; exact two-sided binomial
    p-value on the discordant pairs.  With ~20 scored cases only large
    differences can reach significance — report b and c, not just p.
    """

    def first_rep(rows: list[dict]) -> dict[str, Optional[bool]]:
        return {
            r["case_id"]: r["correct"]
            for r in rows
            if r["rep"] == 1 and r["verified"] and r["correct"] is not None
        }

    a, b_ = first_rep(records_a), first_rep(records_b)
    common = sorted(set(a) & set(b_))
    b = sum(1 for c in common if a[c] and not b_[c])
    c = sum(1 for c in common if not a[c] and b_[c])
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {
        "n_paired": len(common),
        "a_right_b_wrong": b,
        "a_wrong_b_right": c,
        "p_value": round(p, 4),
    }
