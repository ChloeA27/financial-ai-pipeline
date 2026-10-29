"""
--concurrency must actually bound the number of in-flight pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest

from src import main as main_module


class _Probe:
    """Replaces run_pipeline; records the peak number of concurrent calls."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def __call__(self, file_path: str) -> dict:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"file_path": file_path, "validation_passed": True}


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _Probe:
    p = _Probe()
    monkeypatch.setattr(main_module, "run_pipeline", p)
    return p


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_gather_respects_limit(probe: _Probe, limit: int) -> None:
    results = await main_module._gather_with_semaphore(
        [f"f{i}.txt" for i in range(8)], concurrency=limit
    )
    assert len(results) == 8
    assert probe.peak == limit


async def test_batch_passes_concurrency_through(
    probe: _Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Repo:
        async def save_extraction(self, state: dict) -> None:
            return None

    states: list[dict[str, Any]] = [{"file_path": f"f{i}.txt"} for i in range(6)]
    await main_module._process_batch_concurrent(states, _Repo(), concurrency=2)  # type: ignore[arg-type]
    assert probe.peak == 2


def test_positive_int_rejects_zero_and_negative() -> None:
    assert main_module._positive_int("4") == 4
    for bad in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            main_module._positive_int(bad)


async def test_cli_forwards_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """End-to-end through main(): `--concurrency 2` reaches the semaphore."""
    for i in range(5):
        (tmp_path / f"d{i}.txt").write_text("x")

    seen: dict[str, Any] = {}

    async def fake_batch(states, repo, concurrency=5):
        seen["concurrency"] = concurrency
        return []

    async def noop() -> None:
        return None

    monkeypatch.setattr(main_module, "_process_batch_concurrent", fake_batch)
    monkeypatch.setattr(main_module, "init_db", noop)
    monkeypatch.setattr(
        "sys.argv", ["main", "--dir", str(tmp_path), "--concurrency", "2"]
    )
    await main_module.main()
    assert seen["concurrency"] == 2
