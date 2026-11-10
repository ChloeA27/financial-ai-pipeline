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

What is stored is the source file's **bytes, unchanged** (same encoding,
same line endings), and the key is the SHA-256 of those bytes — so it matches
``sha256sum`` of the original file.  Decoding to text for the LLM happens
afterwards, in the Reader.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Union

import aiofiles

from src.config import settings


def raw_uri_for(source_hash: str) -> str:
    """Location of the stored text for a content hash."""
    return str(Path(settings.raw_store_path) / f"{source_hash}.txt")


async def store_raw(source_hash: str, data: Union[bytes, str]) -> str:
    """
    Write the content to the raw layer if it is not there yet; return its URI.

    ``data`` should be the original file bytes; ``str`` is accepted (UTF-8
    encoded) for states that were not built from a file.

    Write-once: an existing object is never overwritten.  The write goes to a
    temporary file first and is renamed into place, so a crash cannot leave a
    truncated object under the final name.
    """
    uri = raw_uri_for(source_hash)
    path = Path(uri)
    if path.exists():
        return uri

    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per call: concurrent writers of the same content must not share
    # a temporary file (they would rename it away from each other).
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    if isinstance(data, str):
        data = data.encode("utf-8")
    async with aiofiles.open(tmp, "wb") as fh:
        await fh.write(data)
    if path.exists():  # another writer won the race; same hash => same bytes
        tmp.unlink()
    else:
        os.replace(tmp, path)
    return uri


async def read_raw_bytes(raw_uri: str) -> bytes:
    """Read a stored object back exactly as it was written."""
    async with aiofiles.open(raw_uri, "rb") as fh:
        return await fh.read()


async def read_raw(raw_uri: str) -> str:
    """Read a stored object back as UTF-8 text (convenience for audits)."""
    return (await read_raw_bytes(raw_uri)).decode("utf-8")
