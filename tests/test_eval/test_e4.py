"""E4 scoring: classification by E1 rules, outcome by outcome_of, per label scope."""

from __future__ import annotations

from pathlib import Path

from evals.evaluators import e1, e4


def _case(cid: str, labels: list[str], verified: bool = True) -> e1.Case:
    return e1.Case(
        case_id=cid,
        split="dev",
        group="typical",
        adsh="x",
        input_file=Path("x.txt"),
        verified=verified,
        labels=labels,
    )


def _stored(doc_type: str) -> dict:
    return {
        "classification_passed": True,
        "doc_type": doc_type,
        "validation_passed": True,
        "error": None,
        "extracted_data": {"doc_id": "d", "acquirer": "A", "target": None},
        "step_runs": [
            {
                "stage": "classify",
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "latency_ms": 5,
            },
            {
                "stage": "extract",
                "prompt_tokens": 200,
                "completion_tokens": 20,
                "latency_ms": 7,
            },
        ],
    }


def _review(doc_type: str, code: str = "TARGET.NON_EMPTY") -> dict:
    return {
        "classification_passed": True,
        "doc_type": doc_type,
        "validation_passed": False,
        "error": "stop",
        "failure": {"stage": "validate", "code": code, "category": "missing_info"},
        "step_runs": [{"stage": "extract"}],
    }


def test_accepted_label_counts_even_when_extraction_fails() -> None:
    r = e4.record(_case("A", ["mna"]), _review("M&A"))
    assert r["predicted"] == "mna" and r["classification_correct"] is True
    assert r["outcome"] == "human_review" and r["error_code"] == "TARGET.NON_EMPTY"


def test_record_counts_fields_tokens_and_attempts() -> None:
    r = e4.record(_case("A", ["mna"]), _stored("M&A"))
    assert (r["fields_filled"], r["fields_total"]) == (1, 2)  # doc_id excluded
    assert r["prompt_tokens"] == 300 and r["extract_attempts"] == 1


def test_mgmt_change_only_is_gold_none() -> None:
    assert e4.gold_label(_case("A", ["mgmt_change"])) == "none"
    assert e4.gold_label(_case("B", ["mna", "mgmt_change"])) == "mna"


def test_summary_separates_verified_and_silver() -> None:
    records = [
        e4.record(_case("A", ["mna"], True), _stored("M&A")),
        e4.record(_case("B", ["dividend"], False), _review("Dividend", "X")),
        e4.record(
            _case("C", ["none"], False),
            {
                "classification_passed": True,
                "doc_type": "none",
                "error": "x",
                "failure": {
                    "stage": "classify",
                    "code": "UNSUPPORTED_DOC_TYPE",
                    "category": "out_of_scope",
                },
            },
        ),
    ]
    s = e4.summarize(records)
    assert s["verified"]["n"] == 1 and s["silver"]["n"] == 2 and s["all"]["n"] == 3
    assert s["all"]["outcomes"] == {
        "stored": 1,
        "out_of_scope": 1,
        "human_review": 1,
        "dlq": 0,
    }
    assert s["all"]["outcomes_by_gold"]["dividend"]["human_review"] == 1
    assert s["all"]["supported_stored_rate"] == 0.5
    assert s["silver"]["reasons"] == [
        {"outcome": "human_review", "stage": "validate", "code": "X", "count": 1}
    ]


def test_api_outage_is_not_scored_as_a_classification_miss() -> None:
    state = {
        "error": "x",
        "failure": {
            "stage": "classify",
            "code": "LLM_CONFIG_ERROR",
            "category": "config_error",
        },
    }
    r = e4.record(_case("A", ["mna"]), state)
    assert r["classification_correct"] is None and r["outcome"] == "dlq"
