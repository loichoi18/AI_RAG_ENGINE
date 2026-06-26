"""Deterministic test doubles for the ingestion and retrieval layers."""
from __future__ import annotations
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any
from ingestion.base import Embedder
from models.domain import Chunk, RetrievalResult
from retrieval.access import acl_allows
from storage.base import VectorStore


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


class FakeWordTokenizer:
    def __init__(self) -> None:
        self._id_to_word: list[str] = []; self._word_to_id: dict[str, int] = {}
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._word_to_id:
                self._word_to_id[word] = len(self._id_to_word); self._id_to_word.append(word)
            ids.append(self._word_to_id[word])
        return ids
    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._id_to_word[t] for t in tokens)
    def count(self, text: str) -> int:
        return len(text.split())


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim] += 1.0
        return vec
    @property
    def dimension(self) -> int:
        return self._dim


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.points: dict[str, tuple[Chunk, list[float]]] = {}
        self.deleted: list[str] = []; self.last_search: dict[str, Any] = {}
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.chunk_id] = (chunk, list(vector))
    def search(self, query_vector: Sequence[float], top_k: int, acl: Sequence[str] | None = None,
               metadata_filter: Mapping[str, Any] | None = None,
               score_threshold: float | None = None) -> list[RetrievalResult]:
        self.last_search = {"top_k": top_k, "acl": acl, "metadata_filter": metadata_filter,
                            "score_threshold": score_threshold}
        scored: list[RetrievalResult] = []
        for chunk, vector in self.points.values():
            if not acl_allows(chunk.acl, acl):
                continue
            if metadata_filter and not self._matches(chunk, metadata_filter):
                continue
            score = _cosine(query_vector, vector)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(RetrievalResult(chunk=chunk, score=score, retriever_name="vector_store"))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
    def delete(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self.points.pop(cid, None); self.deleted.append(cid)
    @staticmethod
    def _matches(chunk: Chunk, metadata_filter: Mapping[str, Any]) -> bool:
        payload = {"document_id": chunk.document_id, "chunk_id": chunk.chunk_id}
        return all(payload.get(key) == value for key, value in metadata_filter.items())
