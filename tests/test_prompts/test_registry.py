"""Prompt registry: templates load verbatim, render strictly, fail loudly."""

from __future__ import annotations

import pytest

from src.prompts import available_prompts, load_prompt
from src.prompts.registry import TEMPLATES_DIR


def _raw(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.md").read_text(encoding="utf-8")


def test_every_template_loads_and_is_not_empty() -> None:
    assert available_prompts() == [
        "classifier_system",
        "extractor_default_system",
        "extractor_dividend_system",
        "extractor_ma_system",
    ]
    for name in available_prompts():
        assert load_prompt(name).strip()


def test_text_is_the_file_minus_its_final_newline() -> None:
    # Moving a prompt into a file must not change a byte that is sent.
    raw = _raw("classifier_system")
    assert raw.endswith("\n")
    assert load_prompt("classifier_system") == raw[:-1]


def test_literal_dollar_is_kept_when_no_params_are_passed() -> None:
    assert "no $ signs" in load_prompt("extractor_ma_system")


def test_placeholders_render_and_a_missing_value_is_an_error() -> None:
    assert "this Earnings announcement" in load_prompt(
        "extractor_default_system", doc_type="Earnings"
    )
    with pytest.raises(KeyError):
        load_prompt("extractor_default_system", other="x")


def test_unknown_prompt_names_the_available_ones() -> None:
    with pytest.raises(KeyError, match="classifier_system"):
        load_prompt("does_not_exist")


def test_nodes_send_the_template_text() -> None:
    from src.nodes.classifier import SYSTEM_PROMPT
    from src.nodes.extractor import EXTRACTOR_PROMPTS, _get_default_prompt

    assert SYSTEM_PROMPT == load_prompt("classifier_system")
    assert EXTRACTOR_PROMPTS["Dividend"] == load_prompt("extractor_dividend_system")
    assert _get_default_prompt("Foo").startswith(
        "Extract all structured financial data from this Foo announcement."
    )
