"""
Reader Node — async file I/O that bootstraps the LangGraph pipeline.

Reads a raw text file from disk, initialises a PipelineState dict,
and sends it into the graph for classification + extraction.
"""

from __future__ import annotations

import os
from pathlib import Path

import aiofiles

from src.config import settings
from src.state.pipeline_state import PipelineState


def _init_pipeline_state(file_path: str, raw_content: str) -> PipelineState:
    """Construct a fresh PipelineState for a single document."""
    return PipelineState(
        file_path=file_path,
        raw_content=raw_content,
        doc_type=None,
        extracted_data=None,
        validation_passed=None,
        validation_report=None,
        correction_logs=[],
        retry_count=0,
        max_retries=settings.max_retries,
        error=None,
    )


async def read_single_file(file_path: str) -> PipelineState:
    """
    Read one raw text file asynchronously and return its PipelineState.

    Args:
        file_path: Absolute or relative path to the `.txt` announcement file.

    Returns:
        A fully initialised PipelineState with ``raw_content`` populated.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    async with aiofiles.open(path, mode="r", encoding="utf-8") as fh:
        raw_content = await fh.read()

    return _init_pipeline_state(str(path), raw_content)


async def read_directory(raw_dir: str = "raw_data/sample") -> list[PipelineState]:
    """
    Scan a directory for ``.txt`` files and read every one concurrently.

    Uses ``asyncio.gather`` for parallel file I/O — crucial for processing
    hundreds of announcements in production.

    Args:
        raw_dir: Directory containing raw announcement text files
                 (relative path is resolved from the project root).

    Returns:
        A list of PipelineState dicts, one per file.
    """
    base = Path(raw_dir).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Raw data directory not found: {base}")

    txt_files = sorted(base.glob("*.txt"))

    import asyncio

    states = await asyncio.gather(
        *(read_single_file(str(f)) for f in txt_files),
        return_exceptions=True,
    )

    results: list[PipelineState] = []
    for fpath, res in zip(txt_files, states):
        if isinstance(res, Exception):
            # Log and skip — don't crash the whole batch
            import logging

            logging.warning("Skipping %s: %s", fpath, res)
            continue
        results.append(res)

    return results


# ── LangGraph Node wrapper ──────────────────────────────────────────────


async def reader_node(state: PipelineState) -> dict:
    """
    LangGraph Node: ensures the file is read and state is populated.

    This function is designed to be called **after** the initial state has
    been seeded (e.g. via ``read_single_file``).  It acts as an idempotent
    safeguard — if ``raw_content`` is already present, it passes through.

    Args:
        state: Current pipeline state (may already be populated).

    Returns:
        A dict of state updates for LangGraph to merge.
    """
    if state.get("error"):
        return {}  # Short-circuit on prior error

    if state.get("raw_content"):
        return {}  # Already populated — no-op

    file_path = state.get("file_path", "")
    if not file_path:
        return {"error": "reader_node: no file_path in state"}

    try:
        new_state = await read_single_file(file_path)
        return {
            "file_path": new_state["file_path"],
            "raw_content": new_state["raw_content"],
        }
    except FileNotFoundError as exc:
        return {"error": f"reader_node: {exc}"}
