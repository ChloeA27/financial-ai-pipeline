"""Content hashing shared by ingestion (Reader) and persistence (Repository)."""

from __future__ import annotations

import hashlib
from typing import Union


def content_hash(data: Union[bytes, str]) -> str:
    """
    SHA-256 hex digest.

    The pipeline keys documents on the hash of the source file's *bytes*;
    ``str`` input is UTF-8 encoded first (identical result for UTF-8 files
    with LF line endings).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
