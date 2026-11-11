"""
E1 scorer self-checks (design notes D19, "上线前的自检").

The real classification graph runs end to end; only the LLM is replaced by a
scripted fake. The records then go through the runner's ``_record`` and
``e1.summarize`` exactly as in a real run. If the scoring code is right:

  * gold used as the model's output       → accuracy 100 %
  * a constant ``none`` output            → accuracy = share of cases where none is acceptable
  * an API failure                        → counted as ``error``, never as a wrong answer
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from evals.evaluators import e1

_SCRIPT = Path(__file__).resolve().parents[2] / "evals" / "run_e1.py"
_spec = importlib.util.spec_from_file_location("run_e1_sanity", _SCRIPT)
run_e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_e1)  # type: ignore[union-attr]

# marker → (document text, gold labels, ambiguous, acceptable, verified, oracle answer)
# The oracle answer is the projected gold in the classifier's vocabulary.
QUOTES = {
    "div": "The Board of Directors declared a quarterly cash dividend of $0.25 per share.",
    "mna": "Acme Corp entered into a definitive agreement to acquire Widget Inc for $1 billion.",
    "mgmt": "The Company appointed Jane Roe as Chief Financial Officer effective May 1, 2026.",
    "none": "The Company reported revenue of $12 million for the third quarter of 2026.",
}
CASES: dict[str, dict[str, Any]] = {
    "C1": dict(text=QUOTES["div"], labels=["dividend"], answer=("Dividend", "div")),
    "C2": dict(text=QUOTES["mna"], labels=["mna"], answer=("M&A", "mna")),
    # mgmt_change only → projected gold is none
    "C3": dict(text=QUOTES["mgmt"], labels=["mgmt_change"], answer=("none", None)),
    "C4": dict(text=QUOTES["none"], labels=["none"], answer=("none", None)),
    # mna + dividend → a single-label classifier is right with "unclear"
    "C5": dict(
        text=QUOTES["mna"] + "\n" + QUOTES["div"],
        labels=["mna", "dividend"],
        answer=("unclear", None),
    ),
    # ambiguous: mna or none are both acceptable
    "C6": dict(
        text=QUOTES["mna"],
        labels=["mna"],
        ambiguous=True,
        acceptable=[["mna"], ["none"]],
        answer=("M&A", "mna"),
    ),
    # not human-verified → must not count towards accuracy
    "C7": dict(
        text=QUOTES["div"] + " (unverified)",
        labels=["dividend"],
        verified=False,
        answer=("Dividend", "div"),
    ),
}


def _cases(tmp_path: Path) -> list[e1.Case]:
    out = []
    for cid, spec in CASES.items():
        path = tmp_path / f"{cid}.txt"
        path.write_text(f"FORM 8-K\nItem 8.01 Other Events.\n{spec['text']}\n")
        out.append(
            e1.Case(
                case_id=cid,
                split="dev",
                group="g",
                adsh=cid,
                input_file=path,
                verified=spec.get("verified", True),
                labels=spec["labels"],
                ambiguous=spec.get("ambiguous", False),
                acceptable_labels=spec.get("acceptable", []),
                evidence=[
                    {"label": label, "quote": QUOTES[key]}
                    for label, key in (("mna", "mna"), ("dividend", "div"))
                    if label in spec["labels"]
                ],
            )
        )
    return out


def _fake_llm(answer_for: Callable[[str], Optional[tuple[str, Optional[str]]]]):
    """AsyncOpenAI stand-in: answer_for(user prompt) → (label, quote key) or None = API error."""

    class FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        async def _create(self, *, messages: list[dict[str, str]], **_: Any) -> Any:
            answer = answer_for(messages[-1]["content"])
            if answer is None:
                raise ConnectionError("simulated API outage")
            label, key = answer
            import json

            content = json.dumps(
                {
                    "doc_type": label,
                    "evidence_quote": QUOTES[key] if key else "",
                    "confidence": 0.9,
                    "reasoning": "scripted",
                }
            )
            usage = SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=usage,
            )

        async def close(self) -> None:
            return None

    return FakeOpenAI


def _case_in(prompt: str) -> str:
    # The longest matching text wins (C5 contains C2's and C1's sentences).
    hits = [cid for cid, spec in CASES.items() if spec["text"] in prompt]
    return max(hits, key=lambda cid: len(CASES[cid]["text"]))


async def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer_for) -> dict:
    from src.nodes.reader import read_single_file
    from src.workflows.graph import build_classification_graph

    monkeypatch.setattr("src.llm.client.AsyncOpenAI", _fake_llm(answer_for))
    graph = build_classification_graph()
    records = []
    for case in _cases(tmp_path):
        final = await graph.ainvoke(await read_single_file(str(case.input_file)))
        records.append(run_e1._record(case, "head", 1, final))
    return {"records": records, "summary": e1.summarize(records)}


async def test_gold_as_output_scores_100_percent(tmp_path, monkeypatch) -> None:
    out = await _run(tmp_path, monkeypatch, lambda p: CASES[_case_in(p)]["answer"])
    s = out["summary"]
    assert s["n_scored_verified"] == 6  # C7 is not verified
    assert s["accuracy_verified"] == 1.0
    assert s["per_label"]["mna"]["precision"] == 1.0
    assert s["none_specificity"] == 1.0
    assert s["errors"] == 0
    assert s["input_coverage_verified"] == 1.0  # gold evidence sits in the window


async def test_gold_as_output_has_perfect_per_label_recall(
    tmp_path, monkeypatch
) -> None:
    out = await _run(tmp_path, monkeypatch, lambda p: CASES[_case_in(p)]["answer"])
    s = out["summary"]
    # accepted "unclear" (C5 mna+dividend) is a hand-off, not a miss
    assert s["per_label"]["mna"]["recall"] == 1.0
    assert s["per_label"]["dividend"]["recall"] == 1.0
    assert s["unclear_accepted"] == 1


async def test_constant_none_scores_only_where_none_is_acceptable(
    tmp_path, monkeypatch
) -> None:
    out = await _run(tmp_path, monkeypatch, lambda p: ("none", None))
    s = out["summary"]
    # right: C3 (mgmt only → none), C4 (none), C6 (ambiguous, none acceptable)
    # wrong: C1, C2, C5
    assert s["accuracy_verified"] == 0.5
    assert s["per_label"]["mna"]["recall"] == 0.0
    assert s["per_label"]["dividend"]["recall"] == 0.0
    assert s["none_specificity"] == 1.0


async def test_api_failure_is_an_error_not_a_wrong_answer(
    tmp_path, monkeypatch
) -> None:
    def answer(prompt: str):
        cid = _case_in(prompt)
        return None if cid == "C1" else CASES[cid]["answer"]

    out = await _run(tmp_path, monkeypatch, answer)
    c1 = next(r for r in out["records"] if r["case_id"] == "C1")
    assert c1["predicted"] == "error"
    assert c1["correct"] is None
    s = out["summary"]
    assert s["errors"] == 1
    assert s["n_scored_verified"] == 5  # C1 left out, not counted as wrong
    assert s["accuracy_verified"] == 1.0
