"""E1 scoring rules: projection onto the supported types, correctness, metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.evaluators import e1


def _case(labels, *, ambiguous=False, acceptable=None, verified=True, group="typical"):
    return e1.Case(
        case_id="E1-001",
        split="dev",
        group=group,
        adsh="x",
        input_file=Path("x.txt"),
        verified=verified,
        labels=labels,
        ambiguous=ambiguous,
        acceptable_labels=acceptable or [],
    )


class TestProjection:
    @pytest.mark.parametrize(
        "labels, projected",
        [
            (["mna"], ("mna",)),
            (["mgmt_change"], ("none",)),  # decided: mgmt_change-only counts as none
            (["mna", "mgmt_change"], ("mna",)),
            (["dividend", "mna"], ("dividend", "mna")),
            (["none"], ("none",)),
        ],
    )
    def test_gold_is_projected_onto_supported_types(self, labels, projected) -> None:
        assert e1.project(labels) == projected


class TestCorrectness:
    def test_exact_single_label(self) -> None:
        exp = e1.expected_for(_case(["dividend"]))
        assert e1.is_correct("dividend", exp) is True
        assert e1.is_correct("mna", exp) is False
        assert e1.is_correct("unclear", exp) is False

    def test_mgmt_change_document_expects_none(self) -> None:
        exp = e1.expected_for(_case(["mgmt_change"]))
        assert e1.is_correct("none", exp) is True

    def test_multi_label_accepts_unclear(self) -> None:
        exp = e1.expected_for(_case(["mna", "dividend"]))
        assert exp.multi
        assert e1.is_correct("unclear", exp) is True
        assert e1.is_correct("mna", exp) is False

    def test_ambiguous_accepts_any_acceptable_answer_or_unclear(self) -> None:
        exp = e1.expected_for(
            _case(["mna"], ambiguous=True, acceptable=[["mna"], ["none"]])
        )
        assert e1.is_correct("none", exp) is True
        assert e1.is_correct("unclear", exp) is True

    def test_failed_classification_is_wrong_and_errors_are_not_scored(self) -> None:
        exp = e1.expected_for(_case(["mna"]))
        assert e1.is_correct("failed:EVIDENCE_QUOTE_NOT_FOUND", exp) is False
        assert e1.is_correct("error", exp) is None


class TestPredictedFromState:
    @pytest.mark.parametrize(
        "state, expected",
        [
            ({"classification_passed": True, "doc_type": "M&A"}, "mna"),
            ({"classification_passed": True, "doc_type": "Dividend"}, "dividend"),
            (
                {
                    "classification_passed": True,
                    "doc_type": "none",
                    "failure": {
                        "category": "out_of_scope",
                        "code": "UNSUPPORTED_DOC_TYPE",
                    },
                },
                "none",
            ),
            (
                {
                    "classification_passed": True,
                    "doc_type": "unclear",
                    "failure": {
                        "category": "ambiguous",
                        "code": "CLASSIFICATION_UNCLEAR",
                    },
                },
                "unclear",
            ),
            (
                {
                    "classification_passed": False,
                    "doc_type": None,
                    "failure": {
                        "category": "reference",
                        "code": "EVIDENCE_QUOTE_NOT_FOUND",
                    },
                },
                "failed:EVIDENCE_QUOTE_NOT_FOUND",
            ),
            (
                {"failure": {"category": "transient", "code": "LLM_CALL_FAILED"}},
                "error",
            ),
        ],
    )
    def test_mapping(self, state, expected) -> None:
        assert e1.predicted_from_state(state) == expected


def _rec(
    case_id,
    predicted,
    correct,
    *,
    gold=("mna",),
    verified=True,
    rep=1,
    group="typical",
    **kw,
) -> dict[str, Any]:
    base = {
        "case_id": case_id,
        "group": group,
        "rep": rep,
        "verified": verified,
        "ambiguous": False,
        "gold_projected": list(gold),
        "predicted": predicted,
        "correct": correct,
        "coverage": None,
        "classify_attempts": 1,
        "first_quote_pass": None,
        "tokens": 100,
        "latency_ms": 50,
    }
    base.update(kw)
    return base


class TestSummarize:
    def test_accuracy_counts_only_verified_scoreable_rows(self) -> None:
        rows = [
            _rec("a", "mna", True),
            _rec("b", "none", False),
            _rec("c", "mna", True, verified=False),  # draft gold: not scored
            _rec("d", "error", None),  # infrastructure: not scored
        ]
        s = e1.summarize(rows)
        assert s["n_scored_verified"] == 2
        assert s["accuracy_verified"] == 0.5
        assert s["errors"] == 1

    def test_per_label_precision_recall_and_none_specificity(self) -> None:
        rows = [
            _rec("a", "mna", True, gold=("mna",)),
            _rec("b", "mna", False, gold=("none",)),
            _rec("c", "none", True, gold=("none",)),
            _rec("d", "none", False, gold=("dividend",)),
        ]
        s = e1.summarize(rows)
        assert s["per_label"]["mna"] == {"precision": 0.5, "recall": 1.0, "support": 1}
        assert s["per_label"]["dividend"]["recall"] == 0.0
        assert s["none_specificity"] == 0.5
        assert s["confusion_verified"]["none -> mna"] == 1

    def test_accepted_unclear_is_a_hand_off_not_a_miss(self) -> None:
        rows = [
            # mna+dividend answered "unclear": accepted → left out of recall
            _rec("a", "unclear", True, gold=("dividend", "mna")),
            # single-label mna answered "unclear": not accepted → still a miss
            _rec("b", "unclear", False, gold=("mna",)),
            _rec("c", "mna", True, gold=("mna",)),
        ]
        s = e1.summarize(rows)
        assert s["unclear_accepted"] == 1
        assert s["per_label"]["mna"] == {"precision": 1.0, "recall": 0.5, "support": 2}
        assert s["per_label"]["dividend"]["support"] == 0
        assert s["accuracy_verified"] == round(2 / 3, 4)

    def test_failure_and_retry_rates(self) -> None:
        rows = [
            _rec("a", "failed:EVIDENCE_QUOTE_NOT_FOUND", False, classify_attempts=3),
            _rec("b", "mna", True, first_quote_pass=True),
        ]
        s = e1.summarize(rows)
        assert s["classify_failed_rate"] == 0.5
        assert s["failed_by_code"] == {"EVIDENCE_QUOTE_NOT_FOUND": 1}
        assert s["classify_retry_rate"] == 0.5
        assert s["first_attempt_quote_pass_rate"] == 1.0


class TestStability:
    def test_rep_agreement(self) -> None:
        rows = [
            _rec("a", "mna", True, rep=1),
            _rec("a", "mna", True, rep=2),
            _rec("b", "mna", True, rep=1),
            _rec("b", "none", False, rep=2),
        ]
        assert e1.rep_agreement(rows) == 0.5

    def test_mcnemar_counts_discordant_pairs(self) -> None:
        a = [_rec(c, "mna", ok) for c, ok in zip("wxyz", (True, True, True, False))]
        b = [_rec(c, "mna", ok) for c, ok in zip("wxyz", (False, False, True, False))]
        m = e1.mcnemar(a, b)
        assert (m["n_paired"], m["a_right_b_wrong"], m["a_wrong_b_right"]) == (4, 2, 0)
        assert m["p_value"] == 0.5  # 2 discordant pairs can never be significant


class TestLoadCases:
    def test_split_filter_and_verified_flag(self, tmp_path: Path) -> None:
        rows = [
            {
                "case_id": "E1-002",
                "split": "dev",
                "group": "typical",
                "adsh": "a",
                "labels": ["mna"],
                "status": "verified",
            },
            {
                "case_id": "E1-001",
                "split": "dev",
                "group": "blind",
                "adsh": "b",
                "labels": ["none"],
                "status": "draft_unverified",
            },
            {
                "case_id": "E1-003",
                "split": "test",
                "group": "typical",
                "adsh": "c",
                "labels": ["mna"],
                "status": "verified",
            },
        ]
        ds = tmp_path / "ds.jsonl"
        ds.write_text("\n".join(json.dumps(r) for r in rows))
        cases = e1.load_cases(ds, tmp_path, "dev")
        assert [c.case_id for c in cases] == ["E1-001", "E1-002"]
        assert [c.verified for c in cases] == [False, True]
        assert cases[1].input_file == tmp_path / "a.txt"
