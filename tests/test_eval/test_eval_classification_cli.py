"""E1 runner: protocol guards and the no-LLM modes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_classification.py"
_spec = importlib.util.spec_from_file_location("eval_classification", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)  # type: ignore[union-attr]

FILING = (
    "=== 8-K main.htm ===\nFORM 8-K\n"
    "If an emerging growth company, indicate by check mark ... ☐\n"
    "Item 8.01 Other Events.\nSee Exhibit 99.1.\n"
    "=== EX-99.1 ex.htm ===\n"
    + "Boilerplate. " * 400
    + "\nThe Board declared a quarterly cash dividend of $0.25 per share.\n"
)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "a1.txt").write_text(FILING)
    rows = [
        {
            "case_id": "E1-001",
            "split": "dev",
            "group": "long",
            "adsh": "a1",
            "labels": ["dividend"],
            "status": "verified",
            "evidence": [
                {
                    "label": "dividend",
                    "quote": "declared a quarterly cash dividend of $0.25 per share",
                }
            ],
        },
        {
            "case_id": "E1-090",
            "split": "test",
            "group": "typical",
            "adsh": "a1",
            "labels": ["mna"],
            "status": "verified",
        },
    ]
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps(r) for r in rows))
    return ds


def test_test_split_is_sealed(dataset: Path) -> None:
    with pytest.raises(SystemExit):
        cli._parse_args(["--dataset", str(dataset), "--split", "test"])


def test_dataset_defaults_to_e1_dataset_env(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("E1_DATASET", str(dataset))
    assert cli._parse_args([]).dataset == dataset


def test_dataset_read_from_dotenv_file(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text(f"DEEPSEEK_API_KEY=x\nE1_DATASET={dataset}\n")
    monkeypatch.delenv("E1_DATASET", raising=False)
    monkeypatch.setattr(cli, "ENV_FILE", env)
    args = cli._parse_args([])
    assert args.dataset == dataset
    assert args.inputs_dir == dataset.parent / "inputs"


def test_explicit_dataset_beats_env(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("E1_DATASET", str(tmp_path / "other.jsonl"))
    assert cli._parse_args(["--dataset", str(dataset)]).dataset == dataset


def test_missing_dataset_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("E1_DATASET", raising=False)
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / "absent.env")
    with pytest.raises(SystemExit):
        cli._parse_args([])
    monkeypatch.setenv("E1_DATASET", str(tmp_path / "nope.jsonl"))
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_unknown_variant_is_rejected(dataset: Path) -> None:
    with pytest.raises(SystemExit):
        cli._parse_args(["--dataset", str(dataset), "--variants", "head,magic"])


def test_coverage_only_needs_no_llm_and_writes_results(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_llm(*_, **__):
        raise AssertionError("coverage-only must not call the LLM")

    monkeypatch.setattr("src.llm.client.AsyncOpenAI", no_llm)
    out = tmp_path / "results"
    assert (
        cli.main(
            [
                "--dataset",
                str(dataset),
                "--coverage-only",
                "--out",
                str(out),
                "--budget-tokens",
                "1500",
            ]
        )
        == 0
    )

    [run] = list(out.iterdir())
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    # the dividend sentence is deep in the exhibit: head misses it, structured reaches it
    assert summary["verified_positive"]["head"] == 0.0
    assert summary["verified_positive"]["structured"] == 1.0
    assert [c["case_id"] for c in manifest["cases"]] == ["E1-001"]  # test row excluded
    assert manifest["projection_rule"].startswith("gold ∩ {mna, dividend}")
    assert len(manifest["cases"][0]["input_sha256"]) == 64


def test_estimate_prints_calls_and_tokens(dataset: Path, capsys) -> None:
    assert cli.main(["--dataset", str(dataset), "--estimate", "--reps", "2"]) == 0
    out = capsys.readouterr().out
    assert "first-attempt calls: 6" in out  # 1 case x 3 variants x 2 reps


def test_unverified_draft_labels_are_not_written_per_case(
    dataset: Path, tmp_path: Path
) -> None:
    import csv

    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    rows[0]["status"] = "draft_unverified"  # a blind draft
    dataset.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "results"
    cli.main(["--dataset", str(dataset), "--coverage-only", "--out", str(out)])

    [run] = list(out.iterdir())
    [row] = list(csv.DictReader(open(run / "coverage.csv")))
    assert row["gold_projected"] == ""
    assert row["cov_head"] == "" and row["cov_structured"] == ""
