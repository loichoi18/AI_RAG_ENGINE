"""Reciprocal Rank Fusion (RRF).

RRF merges several ranked result lists into one using only **ranks**, not
scores: each item accrues ``1 / (k + rank)`` from every list it appears in, and
items are re-sorted by that sum. Because it ignores the raw score magnitudes, it
sidesteps the incomparable-scale problem of dense (cosine) vs sparse (BM25)
scores and needs no per-corpus normalization — which is exactly why it is the
default fusion method for hybrid retrieval and multi-query merging.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual rank learning methods" (SIGIR 2009). The constant ``k`` (default 60)
damps the influence of very high ranks.
"""

from __future__ import annotations

from collections.abc import Sequence

from models.domain import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
    retriever_name: str = "rrf",
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists into one via RRF.

    Parameters
    ----------
    result_lists:
        Ranked lists (each already sorted best-first) to merge. Items are keyed
        by ``chunk.chunk_id`` so the same chunk found by multiple sources has its
        contributions summed.
    k:
        RRF damping constant.
    top_k:
        If given, truncate the fused output to this many results.
    retriever_name:
        Name stamped on the fused results (e.g. ``"hybrid"``, ``"multi_query"``).

    Returns
    -------
    list[RetrievalResult]
        Fused results ordered by descending RRF score.
    """
    scores: dict[str, float] = {}
    representative: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            representative.setdefault(chunk_id, result)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused = [
        RetrievalResult(
            chunk=representative[chunk_id].chunk,
            score=score,
            retriever_name=retriever_name,
        )
        for chunk_id, score in ordered
    ]
    return fused[:top_k] if top_k is not None else fused
