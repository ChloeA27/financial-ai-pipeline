"""
Raw (bronze) layer — immutable, content-addressed copies of every source text.

Each document's text is written once to ``<raw_store_path>/<sha256>.txt`` and
never modified afterwards.  The file name *is* the content hash, so:

  * identical content is stored once, whatever file path it arrived under;
  * an edited source file gets a new object, and the old version stays
    readable — older ``extractions`` rows (SCD Type 2 history) and DLQ rows
    keep pointing at the exact text they were produced from;
  * moving to object storage later (e.g. S3) only changes how ``raw_uri`` is
    written and read; the key scheme stays the same.

What is stored is the text exactly as the pipeline saw it (decoded as UTF-8
in text mode, so CRLF line endings arrive as LF).  That is the text the hash
and the LLM calls refer to; it is not a byte-for-byte copy of the original file.
"""

from __future__ import annotations

import os
from pathlib import Path

import aiofiles

from src.config import settings


def raw_uri_for(source_hash: str) -> str:
    """Location of the stored text for a content hash."""
    return str(Path(settings.raw_store_path) / f"{source_hash}.txt")


async def store_raw(source_hash: str, text: str) -> str:
    """
    Write the text to the raw layer if it is not there yet; return its URI.

    Write-once: an existing object is never overwritten.  The write goes to a
    temporary file first and is renamed into place, so a crash cannot leave a
    truncated object under the final name.
    """
    uri = raw_uri_for(source_hash)
    path = Path(uri)
    if path.exists():
        return uri

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    async with aiofiles.open(tmp, "w", encoding="utf-8", newline="") as fh:
        await fh.write(text)
    os.replace(tmp, path)
    return uri


async def read_raw(raw_uri: str) -> str:
    """Read a stored text back (used for audits and re-runs)."""
    async with aiofiles.open(raw_uri, "r", encoding="utf-8", newline="") as fh:
        return await fh.read()
