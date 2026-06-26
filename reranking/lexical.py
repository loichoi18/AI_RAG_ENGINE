"""Lexical reranker — a deterministic, offline baseline ``Reranker``.

Scores each candidate by the fraction of query content words it contains,
yielding a normalized [0, 1] score without any model. It is useful as a cheap
baseline to compare the cross-encoder against, and it lets the full
gate→generate pipeline run offline (the confidence gate needs interpretable
[0, 1] scores).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from models.domain import RetrievalResult
from reranking.base import Reranker

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class LexicalReranker(Reranker):
    """Re-scores candidates by query content-word coverage."""

    name = "lexical_reranker"

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score by lexical overlap and return the best ``top_k``."""
        query_words = _words(query)
        if not query_words:
            return list(results)[:top_k]

        rescored = [
            RetrievalResult(
                chunk=r.chunk,
                score=len(query_words & _words(r.chunk.text)) / len(query_words),
                retriever_name=self.name,
            )
            for r in results
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
