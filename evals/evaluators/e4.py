"""
E4 end-to-end funnel: where each real filing ends up when the whole pipeline
runs, scored against the E1 labels.

Pure scoring (no LLM, no I/O). The runner ``evals/run_e4.py`` executes the
pipeline and calls these functions.

Per document: the classification (scored with the E1 rules, ``evals/evaluators/e1.py``),
the final outcome (``src/schemas/outcome.py``: stored / out_of_scope /
human_review / dlq), the failure that ended it, and how many extracted fields
are filled. Field values are NOT scored here (no field gold yet; that is E2,
see ``evals/E2_DESIGN.md``).

Every number is reported for three label scopes, because most E1 labels are
unverified drafts: ``verified`` (human-checked), ``silver`` (draft labels) and ``all``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from evals.evaluators import e1
from src.schemas.outcome import OUTCOMES, outcome_of

_PIPELINE_FIELDS = {"doc_id", "metadata"}
SCOPES = ("verified", "silver", "all")


def classification_of(state: dict[str, Any]) -> str:
    """E1 vocabulary; an accepted label counts even if extraction failed later."""
    if state.get("classification_passed") is True:
        label = e1.PIPELINE_TO_E1.get(state.get("doc_type") or "")
        if label:
            return label
    return e1.predicted_from_state(state)


def gold_label(case: e1.Case) -> str:
    """The projected gold label as one string, e.g. ``mna`` or ``dividend+mna``."""
    return "+".join(e1.project(case.labels))


def record(case: e1.Case, state: dict[str, Any]) -> dict[str, Any]:
    """One row per document: classification, outcome, failure, fields, cost."""
    predicted = classification_of(state)
    failure = state.get("failure") or {}
    steps = state.get("step_runs") or []
    fields = {
        k: v
        for k, v in (state.get("extracted_data") or {}).items()
        if k not in _PIPELINE_FIELDS
    }
    return {
        "case_id": case.case_id,
        "group": case.group,
        "verified": case.verified,
        "gold": gold_label(case),
        "predicted": predicted,
        "classification_correct": e1.is_correct(predicted, e1.expected_for(case)),
        "outcome": outcome_of(state),
        "failed_stage": failure.get("stage"),
        "error_code": failure.get("code"),
        "error_category": failure.get("category"),
        "extract_attempts": sum(1 for s in steps if s.get("stage") == "extract"),
        "fields_filled": sum(v is not None for v in fields.values()),
        "fields_total": len(fields),
        "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in steps),
        "completion_tokens": sum(s.get("completion_tokens") or 0 for s in steps),
        "latency_ms": sum(s.get("latency_ms") or 0 for s in steps),
    }


def _in_scope(r: dict[str, Any], scope: str) -> bool:
    return scope == "all" or (
        r["verified"] if scope == "verified" else not r["verified"]
    )


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per scope: classification accuracy, outcome funnel by gold label, reasons."""
    out: dict[str, Any] = {}
    for scope in SCOPES:
        rows = [r for r in records if _in_scope(r, scope)]
        scored = [r for r in rows if r["classification_correct"] is not None]
        by_gold: dict[str, Counter] = {}
        for r in rows:
            by_gold.setdefault(r["gold"], Counter())[r["outcome"]] += 1
        reasons = Counter(
            (r["outcome"], r["failed_stage"], r["error_code"])
            for r in rows
            if r["outcome"] in ("human_review", "dlq")
        )
        positives = [r for r in rows if r["gold"] in ("mna", "dividend")]
        out[scope] = {
            "n": len(rows),
            "classification_accuracy": _rate(
                sum(r["classification_correct"] for r in scored), len(scored)
            ),
            "classification_scored": len(scored),
            "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES},
            "outcomes_by_gold": {
                g: {o: c[o] for o in OUTCOMES} for g, c in sorted(by_gold.items())
            },
            # Supported filings (single label) that were stored: the touchless
            # share before field accuracy is known.
            "supported_stored_rate": _rate(
                sum(r["outcome"] == "stored" for r in positives), len(positives)
            ),
            "reasons": [
                {"outcome": o, "stage": s, "code": c, "count": n}
                for (o, s, c), n in reasons.most_common()
            ],
            "tokens_per_doc": _rate(
                sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
                len(rows),
            ),
            "latency_ms_per_doc": _rate(sum(r["latency_ms"] for r in rows), len(rows)),
        }
    return out
