"""Abstract interface for the reranking layer.

The :class:`Reranker` is the second stage of a two-stage retrieval funnel: it
re-scores a candidate set produced by first-stage retrieval using a model that
reads the (query, chunk) pair jointly (typically a cross-encoder), yielding
higher precision at the top of the list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from models.domain import RetrievalResult


class Reranker(ABC):
    """Re-orders candidate results by joint (query, chunk) relevance."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score ``results`` for ``query`` and return the best ``top_k``.

        Parameters
        ----------
        query:
            The original query.
        results:
            First-stage candidates to re-score.
        top_k:
            Number of results to keep after reranking.

        Returns
        -------
        list[RetrievalResult]
            Reranked results (length ``<= top_k``) with updated scores.
        """
        raise NotImplementedError
