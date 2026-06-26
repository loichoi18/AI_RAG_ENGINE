"""Multi-query retriever.

Expands one user query into several variants (via a :class:`QueryRewriter`),
retrieves for each with the wrapped retriever, and fuses the result lists with
RRF. Querying the index from multiple angles improves recall when the user's
phrasing differs from the document's, and RRF rewards chunks that surface across
several variants.

It wraps any :class:`Retriever` (typically the hybrid one) and is itself a
:class:`Retriever`, so it composes cleanly with caching.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from models.domain import RetrievalResult
from retrieval.base import Retriever
from retrieval.fusion import reciprocal_rank_fusion
from services.query_rewriting.base import QueryRewriter
from utils.logging import get_logger

logger = get_logger(__name__)


class MultiQueryRetriever(Retriever):
    """Expands a query into variants and fuses their retrievals via RRF."""

    name = "multi_query"

    def __init__(
        self,
        retriever: Retriever,
        rewriter: QueryRewriter,
        rrf_k: int = 60,
        max_queries: int = 3,
        per_query_k: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._rewriter = rewriter
        self._rrf_k = rrf_k
        self._max_queries = max_queries
        self._per_query_k = per_query_k

    def retrieve(
        self,
        query: str,
        top_k: int,
        acl: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        """Expand ``query``, retrieve per variant, and RRF-merge the results."""
        started = time.perf_counter()
        variants = self._rewriter.expand_query(query)[: self._max_queries]
        per_query_k = self._per_query_k or top_k

        result_lists = [
            self._retriever.retrieve(variant, per_query_k, acl) for variant in variants
        ]
        fused = reciprocal_rank_fusion(
            result_lists, k=self._rrf_k, top_k=top_k, retriever_name=self.name
        )
        logger.info(
            "retrieval.multi_query",
            original=query,
            variants=variants,
            variant_count=len(variants),
            fused=len(fused),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return fused
