"""Offline evaluation components (deterministic, no models or services).

Provides a hashing embedder whose cosine similarity tracks lexical overlap, so
the evaluation harness can run end-to-end in CI without downloading the bge model
or running Qdrant. These are real, deterministic components — not test doubles —
intended for reproducible offline benchmarking.
"""

from __future__ import annotations

import hashlib

from ingestion.base import Embedder


class HashingEmbedder(Embedder):
    """Deterministic bag-of-hashed-words embedder.

    Each word increments a bucket chosen by a stable hash, so documents sharing
    vocabulary have high cosine similarity. Useful as a fast, reproducible stand-in
    for a semantic embedder during offline evaluation.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        return vec

    @property
    def dimension(self) -> int:
        return self._dim
