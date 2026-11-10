"""Content hashing shared by ingestion (Reader) and persistence (Repository)."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the text exactly as the pipeline processed it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
