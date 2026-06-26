"""Retrieval metrics computed from ranked relevance flags.

All functions take ``relevances`` — a list of booleans in ranked order, where
``relevances[i]`` is True if the chunk at rank ``i`` is relevant — plus, where
needed, ``total_relevant`` (the number of relevant chunks that exist in the
corpus for the query). Separating the relevance judgement (in
``evaluation.dataset``) from the math keeps these metrics reusable and trivially
testable on hand-built inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevances: Sequence[bool], total_relevant: int, k: int) -> float:
    """Fraction of all relevant chunks that appear in the top ``k``.

    Returns 0.0 when no relevant chunks exist (vacuously, nothing to recall).
    """
    if total_relevant <= 0:
        return 0.0
    hits = sum(1 for rel in list(relevances)[:k] if rel)
    return hits / total_relevant


def precision_at_k(relevances: Sequence[bool], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    top = list(relevances)[:k]
    return sum(1 for rel in top if rel) / k


def reciprocal_rank(relevances: Sequence[bool]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none are relevant."""
    for index, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(relevances: Sequence[bool], k: int) -> float:
    """1.0 if at least one relevant result appears in the top ``k``, else 0.0.

    The coarsest retrieval signal — "did we surface anything useful?" — and a
    useful headline number even when ranking quality varies.
    """
    if k <= 0:
        return 0.0
    return 1.0 if any(list(relevances)[:k]) else 0.0


def dcg_at_k(relevances: Sequence[bool], k: int) -> float:
    """Discounted cumulative gain over binary relevance for the top ``k``."""
    return sum(
        (1.0 if rel else 0.0) / math.log2(rank + 1)
        for rank, rel in enumerate(list(relevances)[:k], start=1)
    )


def ndcg_at_k(relevances: Sequence[bool], total_relevant: int, k: int) -> float:
    """Normalized DCG@k: DCG divided by the ideal DCG.

    The ideal ranking places ``min(total_relevant, k)`` relevant items first.
    Returns 0.0 when no relevant items exist.
    """
    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(relevances, k) / ideal if ideal > 0 else 0.0
