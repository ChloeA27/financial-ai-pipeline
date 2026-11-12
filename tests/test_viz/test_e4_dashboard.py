"""
E4 dashboard: the data layer on a written run directory, and a headless render
of the page and of the multi-page app when the dashboard deps are installed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evals.evaluators import e1, e4
from src.viz import e4_data as d


def _case(cid: str, labels: list[str], verified: bool) -> e1.Case:
    return e1.Case(
        case_id=cid,
        split="dev",
        group="typical",
        adsh="x",
        input_file=Path("x.txt"),
        verified=verified,
        labels=labels,
    )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    records = [
        e4.record(
            _case("E1-001", ["mna"], True),
            {
                "classification_passed": True,
                "doc_type": "M&A",
                "validation_passed": True,
                "error": None,
                "extracted_data": {"acquirer": "A", "target": "B"},
            },
        ),
        e4.record(
            _case("E1-002", ["dividend"], False),
            {
                "classification_passed": True,
                "doc_type": "Dividend",
                "validation_passed": False,
                "error": "x",
                "failure": {
                    "stage": "validate",
                    "code": "RECORD_DATE.DATE_FORMAT",
                    "category": "missing_info",
                },
            },
        ),
    ]
    run = tmp_path / "e4" / "20261112-000000_test"
    (run / "_local").mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "wall_seconds": 1.0,
                "llm_tokens": {"prompt": 10, "completion": 2},
                "scopes": e4.summarize(records),
            }
        )
    )
    (run / "manifest.json").write_text(json.dumps({"model": "m", "cases": 2}))
    with open(run / "_local" / "per_case_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    return run


def test_list_runs_finds_runs_with_a_summary(run_dir: Path) -> None:
    (run_dir.parent / "empty").mkdir()
    assert d.list_runs(run_dir.parent) == [run_dir]
    assert d.list_runs(run_dir.parent / "missing") == []


def test_funnel_rows_cover_every_outcome_per_gold_label(run_dir: Path) -> None:
    scope = d.load_summary(run_dir)["scopes"]["all"]
    rows = d.funnel_rows(scope)
    assert [r["gold"] for r in rows[:4]] == ["mna"] * 4  # M&A first
    stored = [r for r in rows if r["gold"] == "mna" and r["outcome"] == "Stored"][0]
    assert (stored["count"], stored["share"]) == (1, 1.0)
    assert d.kpis(scope)["human_review"] == 1


def test_reasons_and_local_cases(run_dir: Path) -> None:
    scope = d.load_summary(run_dir)["scopes"]["silver"]
    assert d.reason_rows(scope)[0]["code"] == "RECORD_DATE.DATE_FORMAT"
    rows, full = d.load_cases(run_dir)
    assert full and len(rows) == 2 and rows[0]["verified"] is True


def test_page_renders(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("E4_RESULTS_DIR", str(run_dir.parent))
    app = Path(__file__).resolve().parents[2] / "src" / "e4_dashboard.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("E4")
    assert len(at.metric) == 6


def test_multipage_app_renders_e4_by_default(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("E4_RESULTS_DIR", str(run_dir.parent))
    app = Path(__file__).resolve().parents[2] / "src" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("E4")
