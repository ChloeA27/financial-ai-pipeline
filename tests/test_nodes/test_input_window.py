"""Classifier input selection: plain-code strategies, no LLM."""

from __future__ import annotations

import pytest

from src.nodes.classification_validator import normalise_text
from src.nodes.input_window import (
    HEAD_CHARS,
    build_input,
    evidence_in_window,
    split_documents,
    strip_cover_page,
)

COVER = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nFORM 8-K\n"
    + "Commission File Number 1-234\n" * 40
    + "Indicate by check mark whether the registrant is an emerging growth company ☐\n"
    "If an emerging growth company, indicate by check mark if the registrant has "
    "elected not to use the extended transition period. ☐\n"
)
INTRO = "Explanatory Note\nOn May 1, Acme merged with and into Widgets Corp.\n\n"
ITEMS = (
    "Item 2.01 Completion of Acquisition or Disposition of Assets.\n"
    + "Details of the closing. " * 200
    + "\nItem 9.01 Financial Statements and Exhibits.\n(d) Exhibits 99.1 Press release\n"
)
EXHIBIT = (
    "Acme Completes Acquisition of Widgets Corp\n\n"
    + "Background paragraph. " * 300
    + "\n\nThe board also declared a quarterly dividend of $0.10 per share.\n"
)
FILING = (
    f"=== 8-K main.htm ===\n{COVER}{INTRO}{ITEMS}=== EX-99.1 ex991.htm ===\n{EXHIBIT}"
)


class TestCoverPage:
    def test_cut_after_the_emerging_growth_company_checkbox(self) -> None:
        body = strip_cover_page(COVER + INTRO + ITEMS)
        assert "Commission File Number" not in body
        assert body.startswith("Explanatory Note")  # the note survives

    def test_6k_cut_after_the_40f_checkbox(self) -> None:
        body = strip_cover_page(
            "FORM 6-K\nForm 20-F ☒ Form 40-F ☐\nNutrien declared a dividend.\n"
        )
        assert body == "Nutrien declared a dividend.\n"

    def test_fallback_to_the_first_item_header(self) -> None:
        body = strip_cover_page("cover text\nItem 8.01 Other Events.\nNews.")
        assert body.startswith("Item 8.01")

    def test_unrecognised_text_is_left_alone(self) -> None:
        assert strip_cover_page("just a press release") == "just a press release"


class TestSplitDocuments:
    def test_markers_split_main_document_and_exhibits(self) -> None:
        docs = split_documents(FILING)
        assert [d for d, _ in docs] == ["8-K", "EX-99.1"]

    def test_plain_text_is_one_document(self) -> None:
        assert split_documents("no markers") == [("DOC", "no markers")]


class TestStrategies:
    def test_head_is_the_original_window(self) -> None:
        w = build_input(FILING, "head")
        assert w.text == FILING[:HEAD_CHARS]
        assert w.truncated and w.chars_total == len(FILING)

    def test_structured_reaches_every_section_within_budget(self) -> None:
        w = build_input(FILING, "structured", budget_tokens=1000)
        text = normalise_text(w.text)

        assert "commission file number" not in text  # cover dropped
        assert "merged with and into widgets corp" in text  # intro kept
        assert "acme completes acquisition of widgets corp" in text  # exhibit headline
        assert w.sections == ["8-K intro", "8-K Item 2.01", "8-K Item 9.01", "EX-99.1"]
        assert w.chars_sent <= 4000 + 200  # budget plus section labels / cut marks
        assert w.truncated

    def test_item_901_contributes_only_its_header(self) -> None:
        w = build_input(FILING, "structured", budget_tokens=3000)
        assert "(d) Exhibits 99.1" not in w.text

    def test_keyword_adds_matching_paragraphs_after_the_lead(self) -> None:
        w = build_input(FILING, "keyword", budget_tokens=1500)
        assert "declared a quarterly dividend of $0.10 per share" in w.text
        assert "keyword paragraphs" in w.sections
        assert w.chars_sent <= 6000

    def test_short_document_fits_untruncated(self) -> None:
        w = build_input(
            "Item 8.01 Other Events.\nAcme declared a dividend.", "structured"
        )
        assert not w.truncated
        assert "Acme declared a dividend." in w.text

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_input("x", "random")

    def test_stats_describe_the_window(self) -> None:
        stats = build_input(FILING, "structured", 1000).stats()
        assert stats["strategy"] == "structured"
        assert stats["chars_sent"] < stats["chars_total"]
        assert set(stats) == {
            "strategy",
            "budget_chars",
            "chars_total",
            "chars_sent",
            "sections",
            "truncated",
        }


class TestCoverage:
    def test_all_quotes_must_be_inside(self) -> None:
        win = "Acme declared a dividend. Acme also acquired Widgets."
        assert evidence_in_window(["acme declared a dividend"], win, normalise_text)
        assert not evidence_in_window(
            ["acme declared a dividend", "not in the window"], win, normalise_text
        )

    def test_no_evidence_means_not_applicable(self) -> None:
        assert evidence_in_window([], "anything", normalise_text) is None
