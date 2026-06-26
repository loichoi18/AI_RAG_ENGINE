"""Sparse lexical retriever using BM25 (Okapi).

BM25 ranks by term overlap, capturing exact tokens that dense embeddings blur
together — error codes, identifiers, rare acronyms common in internal docs. The
index is held in memory (``rank_bm25``) and built from the chunk corpus, which
is scrolled out of the vector store so there is a single source of truth.

Filtering (ACL and explicit document ids) is applied to candidates before
ranking is returned, mirroring the dense path's pre-ranking filter semantics.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from models.domain import Chunk, RetrievalResult
from retrieval.access import acl_allows
from retrieval.base import Retriever
from utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    """In-memory BM25 retriever over a chunk corpus."""

    name = "sparse"

    def __init__(self, chunks: Sequence[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """(Re)build the BM25 index from ``chunks``."""
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        # rank_bm25 requires a non-empty corpus; guard the empty case.
        self._bm25 = BM25Okapi(corpus) if corpus else None
        logger.info("retrieval.bm25_indexed", documents=len(self._chunks))

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        top_k: int,
        acl: Sequence[str] | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-``k`` BM25 matches the caller may access."""
        started = time.perf_counter()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        allowed_docs = set(document_ids) if document_ids is not None else None

        candidates: list[RetrievalResult] = []
        for chunk, score in zip(self._chunks, scores, strict=True):
            if not acl_allows(chunk.acl, acl):
                continue
            if allowed_docs is not None and chunk.document_id not in allowed_docs:
                continue
            candidates.append(
                RetrievalResult(chunk=chunk, score=float(score), retriever_name=self.name)
            )

        candidates.sort(key=lambda r: r.score, reverse=True)
        results = candidates[:top_k]
        logger.info(
            "retrieval.bm25",
            top_k=top_k,
            returned=len(results),
            top_score=results[0].score if results else None,
            acl_scoped=acl is not None,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return results
