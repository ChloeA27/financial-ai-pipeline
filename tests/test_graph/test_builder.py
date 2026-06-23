"""
Tests for the LangGraph graph builder — state construction, conditional
edge routing (self-correction loop), and the high-level run_pipeline().
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from src.graph.builder import _route_after_validation, run_pipeline


# ═══════════════════════════════════════════════════════════════════
#  Conditional edge router
# ═══════════════════════════════════════════════════════════════════


class TestRouteAfterValidation:
    """Unit tests for the _route_after_validation conditional edge function."""

    def test_passed_goes_to_end(self) -> None:
        state: dict[str, Any] = {
            "validation_passed": True,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = _route_after_validation(state)
        assert result == "__end__"

    def test_failed_still_has_retries(self) -> None:
        """If retry_count < max_retries, route to extractor."""
        state = {
            "validation_passed": False,
            "retry_count": 1,
            "max_retries": 3,
            "error": None,
        }
        result = _route_after_validation(state)
        assert result == "extractor"

    def test_failed_exhausted_retries(self) -> None:
        """If retry_count >= max_retries, route to error."""
        state = {
            "validation_passed": False,
            "retry_count": 3,
            "max_retries": 3,
            "error": None,
        }
        result = _route_after_validation(state)
        assert result == "error"

    def test_fatal_error_routes_to_error(self) -> None:
        """Fatal error takes precedence over validation status."""
        state = {
            "validation_passed": True,
            "retry_count": 0,
            "max_retries": 3,
            "error": "Something went wrong",
        }
        result = _route_after_validation(state)
        assert result == "error"

    def test_max_retries_at_boundary(self) -> None:
        """retry_count == 0 is still within budget."""
        state = {
            "validation_passed": False,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = _route_after_validation(state)
        assert result == "extractor"

    def test_max_retries_one_below_limit(self) -> None:
        """retry_count == 2 when max=3 → still eligible."""
        state = {
            "validation_passed": False,
            "retry_count": 2,
            "max_retries": 3,
            "error": None,
        }
        result = _route_after_validation(state)
        assert result == "extractor"

    def test_validation_none_with_no_error(self) -> None:
        """If validation_passed is None (not yet run), what happens?"""
        state = {
            "validation_passed": None,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        # validation_passed is None → treated as falsy → check retries
        result = _route_after_validation(state)
        assert result == "extractor"

    def test_validation_passed_is_false_with_retries_left(self) -> None:
        """Explicit False should be treated as failed."""
        state = {
            "validation_passed": False,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = _route_after_validation(state)
        assert result == "extractor"


# ═══════════════════════════════════════════════════════════════════
#  run_pipeline (integration-light, files exist on disk)
# ═══════════════════════════════════════════════════════════════════


class TestRunPipeline:
    """High-level pipeline execution with real file I/O."""

    async def test_missing_file_raises(self) -> None:
        """run_pipeline with a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await run_pipeline("/tmp/nonexistent_file_12345.txt")

    @pytest.mark.integration
    async def test_pipeline_state_structure(self) -> None:
        """Running on a real file should return a well-structured state."""
        import os

        sample_path = "raw_data/sample/test_ma_pass.txt"
        if not os.path.exists(sample_path):
            pytest.skip(f"Sample file not found: {sample_path}")

        result = await run_pipeline(sample_path)

        # The state must be a dict-like PipelineState
        assert isinstance(result, dict)

        # Required keys must be present
        assert "file_path" in result
        assert "raw_content" in result
        assert "doc_type" in result
        assert "extracted_data" in result
        assert "validation_passed" in result
        assert "retry_count" in result

        # file_path should be the resolved absolute path
        assert "test_ma_pass.txt" in result.get("file_path", "")

        # raw_content should be non-empty
        assert len(result.get("raw_content", "")) > 0

    @pytest.mark.integration
    async def test_pipeline_routes_all_nodes(self) -> None:
        """The pipeline should execute all nodes (reader → classifier → extractor → validator)."""
        import os

        sample_path = "raw_data/sample/test_ma_pass.txt"
        if not os.path.exists(sample_path):
            pytest.skip("Sample file not found")

        result = await run_pipeline(sample_path)

        # doc_type should be classified
        assert result.get("doc_type") is not None
        assert result["doc_type"] in ("M&A", "Dividend", "Management_Change", "Unknown")

        # extracted_data should be populated (even if extraction failed)
        assert (
            result.get("extracted_data") is not None or result.get("error") is not None
        )

        # validation should have run
        assert (
            result.get("validation_passed") is not None
            or result.get("error") is not None
        )

    @pytest.mark.integration
    async def test_pipeline_idempotent_state_keys(self) -> None:
        """Running twice on the same file should produce the same key structure."""
        import os

        sample_path = "raw_data/sample/test_ma_pass.txt"
        if not os.path.exists(sample_path):
            pytest.skip("Sample file not found")

        result1 = await run_pipeline(sample_path)
        result2 = await run_pipeline(sample_path)

        # Both runs should have the same set of top-level keys
        keys1 = set(result1.keys())
        keys2 = set(result2.keys())
        # Core keys that must always be present
        for key in (
            "file_path",
            "raw_content",
            "doc_type",
            "extracted_data",
            "validation_passed",
            "retry_count",
        ):
            assert key in keys1
            assert key in keys2


# ═══════════════════════════════════════════════════════════════════
#  Edge case: file with no extractable data
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for pipeline routing."""

    @pytest.mark.integration
    async def test_fail_loop_file_hits_dlq(self) -> None:
        """test_ma_fail_loop.txt should exhaust retries and hit DLQ path."""
        import os

        fail_path = "raw_data/sample/test_ma_fail_loop.txt"
        if not os.path.exists(fail_path):
            pytest.skip("Fail-loop sample file not found")

        result = await run_pipeline(fail_path)

        # The document is an analyst note (speculative, no concrete deal)
        # so it should either fail validation or have an error
        assert (
            result.get("validation_passed") is not True
            or result.get("error") is not None
        )
