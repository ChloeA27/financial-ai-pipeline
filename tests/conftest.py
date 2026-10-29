"""
Shared fixtures for the Financial AI Pipeline test suite.
"""

from __future__ import annotations

from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid 'unknown marker' warnings."""
    config.addinivalue_line(
        "markers",
        "real_embedder: test that loads the real sentence-transformers model "
        "(needs network on first run). Deselect for a fully offline run: "
        "pytest -m 'not integration and not real_embedder'.",
    )
    config.addinivalue_line(
        "markers",
        "integration: test that requires real LLM API calls or external services. "
        "Skipped in CI by default (run: pytest -m 'not integration').",
    )


# ════════════════════════════════════════════════════════════════
#  Vector-store isolation (applies to EVERY test)
# ════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _isolated_vector_store(
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """
    Keep every test away from the real ``data/chromadb`` and off the network.

    Anything that reaches ChromaDB — directly, or indirectly through
    ``ExtractionRepository`` -> ``save_to_vector_store`` — gets a private
    per-test directory and a deterministic offline embedding function.
    Tests marked ``real_embedder`` keep the real model (still on a private
    directory). The override is cleared afterwards so nothing leaks between
    tests, whatever order they run in.
    """
    from src.rag import chroma_client

    if request.node.get_closest_marker("real_embedder") is None:
        from tests.fake_embedding import HashingEmbeddingFunction

        fake = HashingEmbeddingFunction()
        monkeypatch.setattr(chroma_client, "get_embedding_function", lambda: fake)

    chroma_client.override_chroma_path(str(tmp_path / "chromadb"))
    yield
    chroma_client.override_chroma_path(None)


# ════════════════════════════════════════════════════════════════
#  Sample data fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_manda_extraction() -> dict[str, Any]:
    """A clean M&A extraction dict (as produced by the Extractor)."""
    return {
        "acquirer": "Microsoft Corporation",
        "target": "Activision Blizzard, Inc.",
        "total_value_usd": 68700000000.0,
        "stake_percentage": 100.0,
        "requires_shareholder_approval": True,
        "payment_method": "Cash",
        "announcement_date": "2022-01-18",
        "expected_close_date": "2023-06-30",
    }


@pytest.fixture
def sample_dividend_extraction() -> dict[str, Any]:
    """A clean Dividend extraction dict (as produced by the Extractor)."""
    return {
        "ticker": "AAPL",
        "declaration_date": "2026-04-30",
        "dividend_cash_amount": 0.52,
        "currency": "USD",
        "record_date": "2026-05-12",
        "ex_dividend_date": "2026-05-08",
        "payment_date": "2026-05-21",
        "dividend_type": "Regular Cash",
        "frequency": "Quarterly",
    }


@pytest.fixture
def sample_pipeline_state(sample_manda_extraction: dict[str, Any]) -> dict[str, Any]:
    """A minimal PipelineState dict for M&A as it reaches the Validator."""
    return {
        "file_path": "/fake/path/ma_test.txt",
        "raw_content": "Fake announcement text for testing.",
        "doc_type": "M&A",
        "extracted_data": sample_manda_extraction,
        "validation_passed": None,
        "validation_report": None,
        "correction_logs": [],
        "retry_count": 0,
        "max_retries": 3,
        "error": None,
    }


@pytest.fixture
def dividend_pipeline_state(
    sample_dividend_extraction: dict[str, Any],
) -> dict[str, Any]:
    """A minimal PipelineState dict for Dividend as it reaches the Validator."""
    return {
        "file_path": "/fake/path/dividend_test.txt",
        "raw_content": "Fake dividend announcement text.",
        "doc_type": "Dividend",
        "extracted_data": sample_dividend_extraction,
        "validation_passed": None,
        "validation_report": None,
        "correction_logs": [],
        "retry_count": 0,
        "max_retries": 3,
        "error": None,
    }
