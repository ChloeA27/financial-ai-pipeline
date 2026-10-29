"""--raw-data-dir must reach the pipeline run, not just the existence check."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("evaluate_script_rd", _SCRIPT)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)  # type: ignore[union-attr]


# Sync on purpose: _run_pipeline_for drives its own event loop.
def test_pipeline_runs_against_custom_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "custom_case.txt").write_text("hello")
    seen: dict[str, Any] = {}

    async def fake_run_pipeline(path: str) -> dict:
        seen["path"] = path
        return {"validation_passed": True}

    monkeypatch.setattr(evaluate, "run_pipeline", fake_run_pipeline)
    evaluate._run_pipeline_for("custom_case.txt", tmp_path)

    assert Path(seen["path"]) == tmp_path / "custom_case.txt"


def test_missing_file_in_custom_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="custom_case.txt"):
        evaluate._run_pipeline_for("custom_case.txt", tmp_path)


def test_default_dir_is_unchanged() -> None:
    import inspect

    default = (
        inspect.signature(evaluate._run_pipeline_for).parameters["raw_data_dir"].default
    )
    assert default == evaluate.RAW_DATA_DIR
