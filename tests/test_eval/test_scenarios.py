"""
Scenario evaluator: outcome gate and field gate are scored separately, pending
cases never fail the gate, and an infrastructure failure is inconclusive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.evaluators import scenarios
from evals.evaluators.scenarios import ScenarioCase, load_cases, score_case, summarize

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "scenarios"


def _case(expected: dict, requires: list[str] | None = None) -> ScenarioCase:
    return ScenarioCase(
        case_id="T",
        input="inputs/x.txt",
        purpose="",
        expected=expected,
        requires=requires or [],
    )


def _stored(fields: dict, doc_type: str = "M&A") -> dict:
    return {
        "validation_passed": True,
        "error": None,
        "doc_type": doc_type,
        "extracted_data": fields,
    }


def _failed(stage: str, category: str, code: str = "X", doc_type=None) -> dict:
    return {
        "validation_passed": None,
        "error": "stopped",
        "doc_type": doc_type,
        "failure": {"stage": stage, "code": code, "category": category},
    }


MA = {"acquirer": "Acme Corp", "target": "Widget Inc", "payment_method": "Cash"}


class TestOutcomeGate:
    def test_matching_outcome_and_failure_passes(self) -> None:
        case = _case(
            {
                "outcome": "human_review",
                "failure": {"stage": "validate", "category": "missing_info"},
            }
        )
        r = score_case(case, _failed("validate", "missing_info"))
        assert r["status"] == "pass" and r["actual_outcome"] == "human_review"

    def test_right_outcome_at_the_wrong_stage_fails(self) -> None:
        case = _case(
            {
                "outcome": "human_review",
                "failure": {"stage": "validate", "category": "missing_info"},
            }
        )
        assert score_case(case, _failed("classify", "ambiguous"))["status"] == "fail"

    def test_code_is_checked_only_when_given(self) -> None:
        loose = _case({"outcome": "dlq", "failure": {"category": "input_error"}})
        strict = _case({"outcome": "dlq", "failure": {"code": "DECODE_FAILED"}})
        state = _failed("read", "input_error", code="FILE_NOT_FOUND")
        assert score_case(loose, state)["status"] == "pass"
        assert score_case(strict, state)["status"] == "fail"

    def test_storing_a_document_that_should_not_be_stored_is_a_false_accept(
        self,
    ) -> None:
        case = _case({"outcome": "out_of_scope"})
        r = score_case(case, _stored(MA))
        assert r["status"] == "fail" and r["false_accept"] is True
        assert summarize([r])["false_accepts"] == ["T"]


class TestFieldGate:
    def test_wrong_required_field_fails_the_case(self) -> None:
        case = _case({"outcome": "stored", "doc_type": "M&A", "fields": MA})
        r = score_case(case, _stored(MA | {"target": "Someone Else"}))
        assert r["status"] == "fail"
        assert [f["field"] for f in r["fields"] if not f["passed"]] == ["target"]

    def test_wrong_optional_field_is_only_a_warning(self) -> None:
        case = _case(
            {
                "outcome": "stored",
                "doc_type": "M&A",
                "fields": MA | {"stake_percentage": 100.0},
            }
        )
        r = score_case(case, _stored(MA | {"stake_percentage": 51.0}))
        assert r["status"] == "pass"
        assert r["field_warnings"] == ["stake_percentage"]
        s = summarize([r])
        assert s["gate_passed"] and s["field_warnings"] == {"T": ["stake_percentage"]}
        assert s["field_accuracy"]["stake_percentage"] == {"passed": 0, "total": 1}

    def test_fields_are_not_scored_when_the_outcome_is_wrong(self) -> None:
        case = _case({"outcome": "stored", "doc_type": "M&A", "fields": MA})
        r = score_case(case, _failed("validate", "missing_info"))
        assert r["fields"] == [] and r["status"] == "fail"


class TestPendingAndInconclusive:
    def test_pending_case_is_reported_but_never_fails_the_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(scenarios, "IMPLEMENTED_FEATURES", frozenset())
        case = _case({"outcome": "stored"}, requires=["presence_levels"])
        r = score_case(case, _failed("validate", "missing_info"))
        s = summarize([r])
        assert r["status"] == "fail" and r["gating"] is False
        assert s["gate_passed"] and s["pending"] == {"T": ["presence_levels"]}

    def test_pending_case_that_passes_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(scenarios, "IMPLEMENTED_FEATURES", frozenset())
        case = _case(
            {"outcome": "stored", "doc_type": "M&A", "fields": MA},
            requires=["presence_levels"],
        )
        s = summarize([score_case(case, _stored(MA))])
        assert s["pending_now_passing"] == ["T"]

    def test_implemented_feature_makes_the_case_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            scenarios, "IMPLEMENTED_FEATURES", frozenset({"presence_levels"})
        )
        case = _case({"outcome": "stored"}, requires=["presence_levels"])
        s = summarize([score_case(case, _failed("validate", "missing_info"))])
        assert not s["gate_passed"] and s["failed"] == ["T"]

    def test_api_outage_is_inconclusive_fails_the_gate_and_is_not_scored(self) -> None:
        case = _case({"outcome": "stored"})
        r = score_case(case, _failed("classify", "transient", "LLM_CALL_FAILED"))
        s = summarize([r])
        assert r["status"] == "inconclusive"
        assert not s["gate_passed"] and s["inconclusive"] == ["T"]
        assert s["outcome_scored"] == 0 and s["confusion"] == []


class TestDataset:
    def test_every_case_loads_and_its_input_exists(self) -> None:
        cases = load_cases(SCENARIOS_DIR)
        assert cases
        for c in cases:
            assert (SCENARIOS_DIR / c.input).is_file(), c.case_id
            assert c.synthetic

    def test_every_outcome_is_covered(self) -> None:
        outcomes = {c.expected["outcome"] for c in load_cases(SCENARIOS_DIR)}
        assert outcomes == {"stored", "out_of_scope", "human_review", "dlq"}

    def test_unknown_outcome_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "cases").mkdir()
        (tmp_path / "cases" / "X.json").write_text(
            '{"case_id": "X", "input": "a.txt", "expected": {"outcome": "passed"}}'
        )
        with pytest.raises(ValueError, match="unknown outcome"):
            load_cases(tmp_path)


class TestOverridesAndDiagnostics:
    def test_one_of_accepts_any_spelling_in_the_source(self) -> None:
        from evals.evaluators.scenarios import compare_field

        rule = {
            "mode": "one_of",
            "values": ["Microsoft Corp.", "Microsoft Corporation"],
        }
        assert compare_field(
            "acquirer", "Microsoft Corporation", "Microsoft Corp.", rule
        )[0]
        assert not compare_field("acquirer", "Microsoft", "Microsoft Corp.", rule)[0]

    def test_a_false_accept_records_what_was_stored(self) -> None:
        case = _case({"outcome": "human_review"})
        r = score_case(case, _stored(MA | {"target": "a private company (not named)"}))
        assert r["false_accept"]
        assert r["stored_fields"]["target"] == "a private company (not named)"

    def test_stored_fields_are_not_repeated_for_correct_stores(self) -> None:
        case = _case({"outcome": "stored", "doc_type": "M&A", "fields": MA})
        assert score_case(case, _stored(MA))["stored_fields"] is None
