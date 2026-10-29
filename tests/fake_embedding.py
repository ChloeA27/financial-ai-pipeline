"""
Deterministic, offline stand-in for the sentence-transformers embedding
function, so vector-store tests need no model download.

Texts sharing words land close together (hashed bag-of-words, L2-normalised),
which is enough to exercise retrieval, filtering and top-k logic. It says
nothing about real semantic quality — tests that need that are marked
``real_embedder`` and use the actual model.
"""

from __future__ import annotations

import hashlib
import math
import re

from chromadb import EmbeddingFunction


class HashingEmbeddingFunction(EmbeddingFunction):
    DIM = 64

    def __init__(self) -> None:
        pass

    def __call__(self, input):  # noqa: A002 - signature fixed by chromadb
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in re.findall(r"[a-z0-9&]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name() -> str:
        return "test-hashing-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction()

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]
