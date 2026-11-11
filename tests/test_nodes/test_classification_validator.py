"""
Classification Validator Node: plain-code checks on the classifier output,
retry bookkeeping for the classify loop, and the terminal none / unclear labels.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.nodes.classification_validator import (
    MIN_QUOTE_CHARS,
    classification_validator_node,
    normalise_text,
    quote_in_source,
)
from src.schemas.classification import ALL_LABELS, SUPPORTED_DOC_TYPES
from src.nodes.extractor import EXTRACTOR_MODELS, EXTRACTOR_PROMPTS

SOURCE = (
    "Microsoft Corp. today announced that it has reached a definitive agreement\n"
    "to acquire Activision Blizzard, Inc. for $68.7 billion in an all-cash\n"
    "transaction. The “deal” is subject to regulatory review."
)


def _state(label: str = "M&A", quote: str = "", **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "file_path": "/fake/ma.txt",
        "raw_content": SOURCE,
        "doc_type": label,
        "classification": {
            "label": label,
            "evidence_quote": quote,
            "confidence": 0.9,
            "reasoning": "r",
        },
        "classification_passed": None,
        "classify_retry_count": 0,
        "classify_max_retries": 2,
        "correction_logs": [],
        "retry_count": 0,
        "max_retries": 3,
        "error": None,
        "failure": None,
    }
    state.update(overrides)
    return state


GOOD_QUOTE = "reached a definitive agreement to acquire Activision Blizzard"


class TestRegistryConsistency:
    def test_supported_labels_match_the_extractor_registry(self) -> None:
        assert set(SUPPORTED_DOC_TYPES) == set(EXTRACTOR_MODELS)
        assert set(SUPPORTED_DOC_TYPES) == set(EXTRACTOR_PROMPTS)

    def test_terminal_labels_are_separate(self) -> None:
        assert {"none", "unclear"} <= set(ALL_LABELS)
        assert not {"none", "unclear"} & set(SUPPORTED_DOC_TYPES)


class TestQuoteMatching:
    def test_exact_substring(self) -> None:
        assert quote_in_source(GOOD_QUOTE, SOURCE)

    def test_is_insensitive_to_case_and_line_breaks(self) -> None:
        assert quote_in_source("REACHED A DEFINITIVE AGREEMENT TO ACQUIRE", SOURCE)

    def test_typographic_quotes_match(self) -> None:
        assert quote_in_source('The "deal" is subject to regulatory', SOURCE)

    def test_surrounding_ellipses_and_quotes_are_ignored(self) -> None:
        assert quote_in_source('"... definitive agreement to acquire ..."', SOURCE)

    def test_paraphrase_does_not_match(self) -> None:
        assert not quote_in_source("agreed to buy Activision Blizzard", SOURCE)

    def test_empty_quote_never_matches(self) -> None:
        assert not quote_in_source("   ", SOURCE)

    def test_normalise_collapses_whitespace(self) -> None:
        assert normalise_text("A \n\t B") == "a b"


class TestAccepted:
    async def test_valid_quote_passes_and_continues(self) -> None:
        update = await classification_validator_node(_state(quote=GOOD_QUOTE))
        assert update["classification_passed"] is True
        assert "error" not in update and "failure" not in update
        [step] = update["step_runs"]
        assert (step["stage"], step["outcome"], step["attempt"]) == (
            "classify_validate",
            "passed",
            1,
        )

    async def test_dividend_label_is_checked_the_same_way(self) -> None:
        update = await classification_validator_node(_state("Dividend", GOOD_QUOTE))
        assert update["classification_passed"] is True
        assert "error" not in update

    async def test_skips_when_a_previous_node_failed(self) -> None:
        assert await classification_validator_node(_state(error="boom")) == {}


class TestRejectedAndRetried:
    @pytest.mark.parametrize(
        "quote, summary_tag",
        [
            ("", "[STRUCTURE]"),
            ("too short", "[STRUCTURE]"),
            ("agreed to buy Activision Blizzard outright", "[REFERENCE]"),
        ],
    )
    async def test_bad_quote_spends_a_classify_retry(
        self, quote: str, summary_tag: str
    ) -> None:
        update = await classification_validator_node(_state(quote=quote))

        assert update["classification_passed"] is False
        assert update["classify_retry_count"] == 1
        assert update["doc_type"] is None  # the rejected label must not stick
        assert "error" not in update
        [log] = update["correction_logs"]
        assert log["loop"] == "classify"
        assert summary_tag in log["error_summary"]

    async def test_does_not_touch_the_extraction_budget(self) -> None:
        update = await classification_validator_node(_state(quote=""))
        assert "retry_count" not in update

    async def test_unknown_label_is_rejected(self) -> None:
        state = _state("Management_Change", GOOD_QUOTE)
        update = await classification_validator_node(state)
        assert update["classification_passed"] is False
        assert "Management_Change" in update["correction_logs"][0]["error_summary"]

    async def test_missing_classification_is_rejected(self) -> None:
        update = await classification_validator_node(_state(classification=None))
        assert update["classification_passed"] is False
        assert update["classify_retry_count"] == 1

    @pytest.mark.parametrize(
        "quote, code, category",
        [
            ("", "EVIDENCE_QUOTE_MISSING", "structure"),
            ("too short", "EVIDENCE_QUOTE_TOO_SHORT", "structure"),
            (
                "agreed to buy Activision Blizzard outright",
                "EVIDENCE_QUOTE_NOT_FOUND",
                "reference",
            ),
        ],
    )
    async def test_exhausted_budget_records_stage_code_and_category(
        self, quote: str, code: str, category: str
    ) -> None:
        update = await classification_validator_node(
            _state(quote=quote, classify_retry_count=2)
        )

        assert update["classify_retry_count"] == 2  # clamped
        assert "classification_validator: max retries (2) exhausted" in update["error"]
        failure = update["failure"]
        assert failure["stage"] == "classify_validate"
        assert failure["code"] == code
        assert failure["category"] == category
        assert failure["details"]["failures"][0]["code"] == code

    def test_min_quote_length_is_a_sane_guard(self) -> None:
        assert MIN_QUOTE_CHARS >= 8


class TestTerminalLabels:
    async def test_none_ends_as_out_of_scope(self) -> None:
        update = await classification_validator_node(_state("none"))

        assert update["classification_passed"] is True
        assert "not a supported type" in update["error"]
        assert update["failure"]["code"] == "UNSUPPORTED_DOC_TYPE"
        assert update["failure"]["category"] == "out_of_scope"
        assert update["failure"]["stage"] == "classify"
        assert "correction_logs" not in update  # not a retryable failure

    async def test_unclear_ends_as_ambiguous(self) -> None:
        update = await classification_validator_node(_state("unclear"))

        assert update["failure"]["code"] == "CLASSIFICATION_UNCLEAR"
        assert update["failure"]["category"] == "ambiguous"
        assert "human review" in update["error"]

    @pytest.mark.parametrize("label", ["none", "unclear"])
    async def test_a_quote_is_not_required_for_terminal_labels(
        self, label: str
    ) -> None:
        update = await classification_validator_node(_state(label, quote=""))
        assert update["classification_passed"] is True
        assert "classify_retry_count" not in update


class TestQuoteScopeAndDiagnostics:
    async def test_quote_must_come_from_the_text_the_model_saw(self) -> None:
        # The quote exists in the file, but not in the window that was sent.
        state = _state(quote=GOOD_QUOTE, classifier_input="Cover page only, no event.")
        update = await classification_validator_node(state)
        assert update["classification_passed"] is False
        assert "[REFERENCE]" in update["correction_logs"][0]["error_summary"]

    async def test_without_a_recorded_window_the_raw_text_is_used(self) -> None:
        update = await classification_validator_node(_state(quote=GOOD_QUOTE))
        assert update["classification_passed"] is True

    async def test_three_words_are_too_short_even_if_long_enough(self) -> None:
        state = _state(quote="Microsoft Corp. today", raw_content=SOURCE * 2)
        update = await classification_validator_node(state)
        assert "shorter than" in update["correction_logs"][0]["error_summary"]

    async def test_diagnostics_are_recorded_on_acceptance(self) -> None:
        update = await classification_validator_node(_state(quote=GOOD_QUOTE))
        diag = update["step_runs"][0]["output"]["quote"]
        assert diag["found"] is True
        assert diag["offset"] is not None
        assert diag["longest_match_ratio"] == 1.0
        assert diag["label_keyword_hit"] is True  # "acquire"

    async def test_near_miss_is_measured_but_still_rejected(self) -> None:
        typo = "reached a definitive agreement to acquire Activision Blizard"
        update = await classification_validator_node(_state(quote=typo))
        diag = update["step_runs"][0]["output"]["quote"]
        assert update["classification_passed"] is False
        assert diag["found"] is False
        assert 0.8 < diag["longest_match_ratio"] < 1.0

    async def test_keyword_signal_flags_an_unrelated_quote_without_rejecting(
        self,
    ) -> None:
        quote = "The “deal” is subject to regulatory review"
        update = await classification_validator_node(_state("Dividend", quote))
        assert update["classification_passed"] is True
        assert update["step_runs"][0]["output"]["quote"]["label_keyword_hit"] is False
