"""
Reader Node — ingestion: async file I/O that bootstraps the LangGraph pipeline.

For each file it:
  1. reads the **bytes** and stores them unchanged in the raw (bronze) layer,
     keyed by their SHA-256 — before anything can go wrong with decoding;
  2. decodes them for the LLM: UTF-8 (with or without BOM), falling back to
     Windows-1252; line endings are normalised to ``\n``;
  3. initialises a PipelineState.

A file that cannot be decoded (or is binary) does not disappear: its bytes
are still stored, and its state carries ``error`` + a ``read`` / DECODE_FAILED
failure, so it flows through the graph untouched and lands in the DLQ.
"""

from __future__ import annotations

import codecs
import os
import uuid
from pathlib import Path
from typing import Optional

import aiofiles

from src.config import settings
from src.state.failure import (
    CAT_INPUT_ERROR,
    DECODE_FAILED,
    FILE_NOT_FOUND,
    NO_FILE_PATH,
    STAGE_READ,
    make_failure,
)
from src.state.pipeline_state import PipelineState
from src.storage.raw_store import store_raw
from src.utils.hashing import content_hash


class SourceDecodeError(ValueError):
    """The source bytes are not text this pipeline can read."""


def decode_source(data: bytes) -> tuple[str, str]:
    """
    Decode source bytes to text; return ``(text, encoding_label)``.

    Order: UTF-8 with BOM → UTF-8 → Windows-1252.  Windows-1252 covers the
    common "smart quotes from Word" case; it cannot decode five byte values,
    so most binary data still fails.  NUL bytes are treated as binary.
    The fallback is recorded (``cp1252``) so such documents can be reviewed.
    """
    if b"\x00" in data:
        raise SourceDecodeError("binary content (NUL bytes)")
    if data.startswith(codecs.BOM_UTF8):
        try:
            text, label = data[len(codecs.BOM_UTF8) :].decode("utf-8"), "utf-8-bom"
        except UnicodeDecodeError as exc:
            raise SourceDecodeError(f"invalid UTF-8 after BOM: {exc}") from exc
    else:
        try:
            text, label = data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            try:
                text, label = data.decode("cp1252"), "cp1252"
            except UnicodeDecodeError as exc:
                raise SourceDecodeError(
                    f"neither UTF-8 nor Windows-1252: {exc}"
                ) from exc
    return text.replace("\r\n", "\n").replace("\r", "\n"), label


def _init_pipeline_state(
    file_path: str,
    raw_content: str,
    *,
    source_hash: Optional[str] = None,
    raw_uri: Optional[str] = None,
    source_encoding: Optional[str] = None,
) -> PipelineState:
    """Construct a fresh PipelineState for a single document."""
    return PipelineState(
        file_path=file_path,
        raw_content=raw_content,
        source_hash=source_hash or content_hash(raw_content),
        raw_uri=raw_uri,
        source_encoding=source_encoding,
        run_id=uuid.uuid4().hex,
        doc_type=None,
        classification=None,
        classification_passed=None,
        classify_retry_count=0,
        classify_max_retries=settings.classify_max_retries,
        extracted_data=None,
        validation_passed=None,
        validation_report=None,
        correction_logs=[],
        retry_count=0,
        max_retries=settings.max_retries,
        error=None,
        failure=None,
        step_runs=[],
    )


async def read_single_file(file_path: str) -> PipelineState:
    """
    Read one raw text file asynchronously and return its PipelineState.

    Args:
        file_path: Absolute or relative path to the `.txt` announcement file.

    Returns:
        A fully initialised PipelineState with ``raw_content`` populated —
        or, for undecodable content, a state that already carries ``error``
        and a ``read`` / DECODE_FAILED ``failure`` (its bytes are stored).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    async with aiofiles.open(path, mode="rb") as fh:
        data = await fh.read()

    source_hash = content_hash(data)
    raw_uri = await store_raw(source_hash, data)

    try:
        raw_content, encoding = decode_source(data)
    except SourceDecodeError as exc:
        state = _init_pipeline_state(
            str(path), "", source_hash=source_hash, raw_uri=raw_uri
        )
        message = f"reader: cannot decode {path.name} — {exc}"
        state["error"] = message
        state["failure"] = make_failure(
            STAGE_READ,
            DECODE_FAILED,
            CAT_INPUT_ERROR,
            message,
            details={"size_bytes": len(data)},
        )
        return state

    return _init_pipeline_state(
        str(path),
        raw_content,
        source_hash=source_hash,
        raw_uri=raw_uri,
        source_encoding=encoding,
    )


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
        message = "reader_node: no file_path in state"
        return {
            "error": message,
            "failure": make_failure(STAGE_READ, NO_FILE_PATH, CAT_INPUT_ERROR, message),
        }

    try:
        new_state = await read_single_file(file_path)
        keys = ("file_path", "raw_content", "source_hash", "raw_uri", "source_encoding")
        update = {k: new_state[k] for k in keys}
        if new_state.get("error"):
            update["error"] = new_state["error"]
            update["failure"] = new_state["failure"]
        return update
    except FileNotFoundError as exc:
        message = f"reader_node: {exc}"
        return {
            "error": message,
            "failure": make_failure(
                STAGE_READ, FILE_NOT_FOUND, CAT_INPUT_ERROR, message
            ),
        }
