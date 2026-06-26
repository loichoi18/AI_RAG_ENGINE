"""Query rewriting interface.

Two operations:

* ``rewrite_query`` normalizes a raw user question into a cleaner search query
  (strip interrogatives/politeness, collapse whitespace).
* ``expand_query`` returns *several* query variants to broaden recall; these are
  retrieved independently and fused (multi-query retrieval).

Defining this as an interface lets a heuristic implementation ship now (no
generation dependency) and an LLM-backed implementation drop in at Sprint 4
without touching the retrieval orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class QueryRewriter(ABC):
    """Normalizes and expands queries for retrieval."""

    @abstractmethod
    def rewrite_query(self, query: str) -> str:
        """Return a normalized single-query rewrite of ``query``."""
        raise NotImplementedError

    @abstractmethod
    def expand_query(self, query: str) -> list[str]:
        """Return one or more query variants (including the normalized form)."""
        raise NotImplementedError
