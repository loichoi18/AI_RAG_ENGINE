"""Abstract interface for the retrieval layer.

A single :class:`Retriever` contract is implemented by dense, sparse, and
hybrid retrievers alike. Because they share a signature, the hybrid retriever
can *compose* the others (Composite pattern) and the orchestrator can swap
strategies without code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from models.domain import RetrievalResult


class Retriever(ABC):
    """Returns the most relevant chunks for a query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        acl: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve up to ``top_k`` results for ``query``.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Maximum number of results.
        acl:
            Access-control identifiers; forwarded to the vector store so
            filtering happens before ranking.

        Returns
        -------
        list[RetrievalResult]
            Results ordered by descending relevance, each tagged with the
            originating retriever name.
        """
        raise NotImplementedError
