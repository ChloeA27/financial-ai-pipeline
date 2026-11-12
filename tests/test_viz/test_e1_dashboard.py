"""
E1 dashboard: the data layer on a real (scripted-LLM) run directory, and a
full render of the Streamlit app when the optional dashboard deps exist.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.viz import e1_data as d
from tests.test_nodes.test_extractor import FakeOpenAI

_SCRIPT = Path(__file__).resolve().parents[2] / "evals" / "run_e1.py"
_spec = importlib.util.spec_from_file_location("eval_classification_viz", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)  # type: ignore[union-attr]

FILING = (
    "=== 8-K main.htm ===\nFORM 8-K\n"
    "If an emerging growth company, indicate by check mark ... ☐\n"
    "Item 8.01 Other Events.\nThe Board declared a quarterly cash dividend of $0.25 per share.\n"
)
ANSWER = json.dumps(
    {
        "doc_type": "Dividend",
        "evidence_quote": "The Board declared a quarterly cash dividend of $0.25 per share",
        "confidence": 0.9,
        "reasoning": "dividend declared",
    }
)


class _Fake(FakeOpenAI):
    """Always returns the same valid answer (order-independent under concurrency)."""

    async def _create(self, *, messages, **_):  # type: ignore[override]
        FakeOpenAI.calls.append(messages)
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=ANSWER))],
            usage=SimpleNamespace(
                prompt_tokens=900, completion_tokens=120, total_tokens=1020
            ),
        )


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.config import settings

    for attr in (
        "sqlite_db_path",
        "raw_store_path",
        "chromadb_path",
        "rag_enabled",
        "classify_temperature",
        "classifier_input_tokens",
        "classifier_input_strategy",
        "deepseek_api_key",
    ):
        monkeypatch.setattr(settings, attr, getattr(settings, attr))  # restored after
    settings.deepseek_api_key = "sk-test"
    FakeOpenAI.calls = []
    monkeypatch.setattr("src.llm.client.AsyncOpenAI", _Fake)

    data = tmp_path / "e1"
    (data / "inputs").mkdir(parents=True)
    rows = []
    for i, status in ((1, "verified"), (2, "draft_unverified")):
        (data / "inputs" / f"a{i}.txt").write_text(FILING)
        rows.append(
            {
                "case_id": f"E1-00{i}",
                "split": "dev",
                "group": "typical",
                "adsh": f"a{i}",
                "labels": ["dividend"],
                "status": status,
                "evidence": [
                    {
                        "label": "dividend",
                        "quote": "declared a quarterly cash dividend of $0.25 per share",
                    }
                ],
            }
        )
    ds = data / "ds.jsonl"
    ds.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "results"
    assert (
        cli.main(
            [
                "--dataset",
                str(ds),
                "--reps",
                "1",
                "--out",
                str(out),
                "--max-calls",
                "50",
            ]
        )
        == 0
    )
    [run] = list(out.iterdir())
    return run


def test_data_layer_on_a_real_run_directory(run_dir: Path) -> None:
    assert d.list_runs(run_dir.parent) == [run_dir]
    run = d.load_run(run_dir)
    rows = d.step_rows(run["db"])
    index = d.run_index(run["per_case"])

    outcomes = d.outcomes_by_variant(rows, index)
    assert set(outcomes) == {"head", "structured", "keyword"}
    assert all(c == {"accepted · 1st attempt": 2} for c in outcomes.values())
    assert d.retry_causes(rows, index) == {}
    assert len(d.llm_calls(rows, index)) == 6

    table = d.metric_table(run["summary"])
    assert table[0]["Metric"].startswith("Accuracy")
    assert table[0]["V1 structured"] == "100.0%"

    verified_only = d.selectable_cases(run["per_case"], include_unverified=False)
    assert {c["case_id"] for c in verified_only} == {"E1-001"}
    case = verified_only[0]
    steps = d.trace(rows, case["run_id"])
    assert [s["stage"] for s in steps] == ["classify", "classify_validate"]
    win = d.window_for(run_dir, steps, run["manifest"]["budget_tokens"])
    html, found = d.highlight(
        win["text"], [(steps[0]["output"]["evidence_quote"], "model")]
    )
    assert found == [True] and '<mark class="model">' in html


def test_unverified_answers_are_not_in_per_case(run_dir: Path) -> None:
    run = d.load_run(run_dir)
    unverified = [r for r in run["per_case"] if r["verified"] != "True"]
    assert unverified and all(r["predicted"] == "" for r in unverified)
    assert (run_dir / "_sealed" / "unverified_predictions.csv").exists()


def test_streamlit_app_renders_all_three_views(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("E1_RESULTS_DIR", str(run_dir.parent))
    app = Path(__file__).resolve().parents[2] / "src" / "e1_dashboard.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3
    assert at.title[0].value.startswith("E1")
    assert any("not human-verified" in t.label for t in at.toggle)
