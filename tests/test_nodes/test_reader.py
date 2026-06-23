"""
Tests for the Reader Node — async file I/O, pipeline state initialisation,
and the LangGraph node wrapper.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.nodes.reader import (
    _init_pipeline_state,
    read_single_file,
    read_directory,
    reader_node,
)


class TestInitPipelineState:
    """Unit tests for _init_pipeline_state."""

    def test_returns_correct_keys(self) -> None:
        state = _init_pipeline_state("/tmp/test.txt", "Hello world")
        assert state["file_path"] == "/tmp/test.txt"
        assert state["raw_content"] == "Hello world"
        assert state["doc_type"] is None
        assert state["extracted_data"] is None
        assert state["validation_passed"] is None
        assert state["validation_report"] is None
        assert state["correction_logs"] == []
        assert state["retry_count"] == 0
        assert state["error"] is None
        assert state["max_retries"] >= 1  # from settings

    def test_max_retries_from_settings(self) -> None:
        """max_retries should come from the Settings object."""
        from src.config import settings
        state = _init_pipeline_state("/tmp/test.txt", "content")
        assert state["max_retries"] == settings.max_retries

    def test_correction_logs_is_new_list(self) -> None:
        """Each call should create a new list reference."""
        state1 = _init_pipeline_state("/tmp/a.txt", "a")
        state2 = _init_pipeline_state("/tmp/b.txt", "b")
        assert state1["correction_logs"] is not state2["correction_logs"]


class TestReadSingleFile:
    """Tests for read_single_file."""

    async def test_read_existing_file(self) -> None:
        """Should read file content and return a valid PipelineState."""
        # Use an existing sample file
        sample = "raw_data/sample/test_ma_pass.txt"
        if not os.path.exists(sample):
            pytest.skip(f"Sample file not found: {sample}")

        state = await read_single_file(sample)
        assert state["file_path"] is not None
        assert len(state["raw_content"]) > 0
        assert "AMD" in state["raw_content"] or "acquisition" in state["raw_content"]

    async def test_resolves_absolute_path(self) -> None:
        """file_path in state should be an absolute path."""
        sample = "raw_data/sample/test_ma_pass.txt"
        if not os.path.exists(sample):
            pytest.skip(f"Sample file not found: {sample}")

        state = await read_single_file(sample)
        resolved = Path(sample).resolve()
        assert state["file_path"] == str(resolved)

    async def test_missing_file_raises(self) -> None:
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await read_single_file("/tmp/does_not_exist_xyz.txt")

    async def test_empty_file_returns_empty_content(self) -> None:
        """An empty file should have raw_content == ''."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f_path = f.name

        try:
            state = await read_single_file(f_path)
            assert state["raw_content"] == ""
        finally:
            Path(f_path).unlink(missing_ok=True)


class TestReadDirectory:
    """Tests for read_directory."""

    async def test_reads_all_txt_files(self) -> None:
        """Should return one state per .txt file in the directory."""
        sample_dir = "raw_data/sample"
        if not os.path.isdir(sample_dir):
            pytest.skip(f"Sample directory not found: {sample_dir}")

        states = await read_directory(sample_dir)
        assert len(states) >= 4  # All 4 sample files

        file_names = [Path(s["file_path"]).name for s in states]
        for expected in (
            "test_ma_pass.txt",
            "test_ma_fail_loop.txt",
            "ma_microsoft_activision.txt",
            "dividend_apple_2026.txt",
        ):
            assert expected in file_names

    async def test_all_states_have_raw_content(self) -> None:
        sample_dir = "raw_data/sample"
        if not os.path.isdir(sample_dir):
            pytest.skip(f"Sample directory not found: {sample_dir}")

        states = await read_directory(sample_dir)
        for s in states:
            assert len(s["raw_content"]) > 0, f"Empty content in {s['file_path']}"

    async def test_nonexistent_directory_raises(self) -> None:
        """Non-existent directory should raise NotADirectoryError."""
        with pytest.raises(NotADirectoryError):
            await read_directory("/tmp/does_not_exist_dir_xyz")

    async def test_empty_directory(self) -> None:
        """A directory with no .txt files should return an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .csv file that shouldn't be picked up
            Path(tmpdir, "data.csv").write_text("a,b,c")
            states = await read_directory(tmpdir)
            assert len(states) == 0

    async def test_skips_non_txt_files(self) -> None:
        """Only .txt files should be included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "data.txt").write_text("hello")
            Path(tmpdir, "data.csv").write_text("a,b,c")
            Path(tmpdir, "data.json").write_text("{}")

            states = await read_directory(tmpdir)
            assert len(states) == 1  # Only the .txt file


class TestReaderNode:
    """Tests for the LangGraph reader_node wrapper."""

    async def test_langgraph_node_noop_when_populated(self) -> None:
        """If raw_content already present, reader_node should return {}."""
        state = {
            "file_path": "/tmp/test.txt",
            "raw_content": "Existing content",
            "doc_type": None,
            "extracted_data": None,
            "validation_passed": None,
            "validation_report": None,
            "correction_logs": [],
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = await reader_node(state)
        assert result == {}

    async def test_langgraph_node_reads_file(self) -> None:
        """If raw_content is missing, reader_node should read from disk."""
        sample = "raw_data/sample/test_ma_pass.txt"
        if not os.path.exists(sample):
            pytest.skip(f"Sample file not found: {sample}")

        state = {
            "file_path": str(Path(sample).resolve()),
            "raw_content": "",
            "doc_type": None,
            "extracted_data": None,
            "validation_passed": None,
            "validation_report": None,
            "correction_logs": [],
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = await reader_node(state)
        assert "raw_content" in result
        assert len(result["raw_content"]) > 0

    async def test_langgraph_node_no_file_path(self) -> None:
        """If file_path is missing, reader_node should return an error."""
        state = {
            "file_path": "",
            "raw_content": "",
            "doc_type": None,
            "extracted_data": None,
            "validation_passed": None,
            "validation_report": None,
            "correction_logs": [],
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = await reader_node(state)
        assert "error" in result

    async def test_langgraph_node_missing_file(self) -> None:
        """If file doesn't exist, reader_node should return an error."""
        state = {
            "file_path": "/tmp/does_not_exist_xyz.txt",
            "raw_content": "",
            "doc_type": None,
            "extracted_data": None,
            "validation_passed": None,
            "validation_report": None,
            "correction_logs": [],
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = await reader_node(state)
        assert "error" in result
        assert "FileNotFoundError" in result["error"] or "not found" in result["error"].lower()

    async def test_langgraph_node_short_circuit_on_error(self) -> None:
        """If state already has error, reader_node should skip."""
        state = {
            "file_path": "/tmp/test.txt",
            "raw_content": "",
            "error": "Previous error",
        }
        result = await reader_node(state)
        assert result == {}
