"""Hybrid retriever: dense + sparse fused with Reciprocal Rank Fusion.

Runs the dense and sparse retrievers independently (each over a wider candidate
pool), then fuses their ranked lists with RRF. This combines semantic recall
with lexical precision while sidestepping the dense-vs-BM25 score-scale problem,
because RRF fuses by rank, not magnitude.

It implements the same :class:`Retriever` contract as its components, so it can
itself be wrapped (e.g. by multi-query or caching) without special-casing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from models.domain import RetrievalResult
from retrieval.base import Retriever
from retrieval.dense.dense_retriever import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse.bm25_retriever import BM25Retriever
from utils.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever(Retriever):
    """Fuses dense and sparse retrieval via RRF."""

    name = "hybrid"

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int,
        acl: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve, fuse, and return the top-``k`` hybrid results."""
        started = time.perf_counter()
        dense_results = self._dense.retrieve(query, self._candidate_k, acl)
        sparse_results = self._sparse.retrieve(query, self._candidate_k, acl)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self._rrf_k,
            top_k=top_k,
            retriever_name=self.name,
        )
        logger.info(
            "retrieval.hybrid",
            top_k=top_k,
            dense=len(dense_results),
            sparse=len(sparse_results),
            fused=len(fused),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return fused
