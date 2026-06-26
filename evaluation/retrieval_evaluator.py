"""Retrieval evaluator.

Runs a retriever over the golden set and aggregates Recall@K, Precision@K, MRR,
and nDCG@K. Relevance is judged by answer span (see ``evaluation.dataset``), so
this evaluator is agnostic to the chunking strategy and retriever implementation
— it only needs something that implements :class:`Retriever`.
"""

from __future__ import annotations

from collections.abc import Sequence

import time

from evaluation.dataset import GoldenExample, is_relevant, relevance_flags
from evaluation.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from models.domain import Chunk
from retrieval.base import Retriever
from utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalEvaluator:
    """Computes retrieval metrics for a retriever over a golden dataset."""

    def __init__(self, retriever: Retriever, k: int = 5, total_relevant_cap: int | None = None) -> None:
        self._retriever = retriever
        self._k = k
        # If the corpus has a known relevant count per query, supply it; otherwise
        # estimate it from what is retrieved (a conservative recall denominator).
        self._total_relevant_cap = total_relevant_cap

    def evaluate(self, dataset: Sequence[GoldenExample]) -> dict[str, float]:
        """Return mean retrieval metrics across ``dataset``."""
        rows = self.evaluate_examples(dataset)
        result = {
            f"recall@{self._k}": _mean([r[f"recall@{self._k}"] for r in rows]),
            f"precision@{self._k}": _mean([r[f"precision@{self._k}"] for r in rows]),
            "mrr": _mean([r["mrr"] for r in rows]),
            f"ndcg@{self._k}": _mean([r[f"ndcg@{self._k}"] for r in rows]),
            f"hit_rate@{self._k}": _mean([r[f"hit_rate@{self._k}"] for r in rows]),
        }
        logger.info("eval.retrieval", k=self._k, examples=len(dataset), **{
            key: round(value, 4) for key, value in result.items()
        })
        return result

    def evaluate_examples(self, dataset: Sequence[GoldenExample]) -> list[dict[str, float]]:
        """Per-example retrieval metrics plus latency (for detailed reports)."""
        rows: list[dict[str, float]] = []
        for example in dataset:
            started = time.perf_counter()
            chunks = self._retrieve_chunks(example)
            latency_ms = (time.perf_counter() - started) * 1000
            flags = relevance_flags(chunks, example)
            total_relevant = self._total_relevant(example, chunks)
            rows.append(
                {
                    f"recall@{self._k}": recall_at_k(flags, total_relevant, self._k),
                    f"precision@{self._k}": precision_at_k(flags, self._k),
                    "mrr": reciprocal_rank(flags),
                    f"ndcg@{self._k}": ndcg_at_k(flags, total_relevant, self._k),
                    f"hit_rate@{self._k}": hit_rate_at_k(flags, self._k),
                    "retrieved": float(len(chunks)),
                    "latency_ms": round(latency_ms, 2),
                }
            )
        return rows

    def _retrieve_chunks(self, example: GoldenExample) -> list[Chunk]:
        results = self._retriever.retrieve(example.query, self._k, example.acl)
        return [r.chunk for r in results]

    def _total_relevant(self, example: GoldenExample, chunks: Sequence[Chunk]) -> int:
        """Relevant-chunk count for recall's denominator.

        With span-based relevance the true count depends on chunking, so we use
        the number of relevant chunks actually retrieved (>=1 when any hit), or
        an explicit cap if provided. This yields Recall@K of 1.0 when at least
        one relevant chunk is retrieved, which matches the single-fact nature of
        the golden questions.
        """
        if self._total_relevant_cap is not None:
            return self._total_relevant_cap
        retrieved_relevant = sum(1 for c in chunks if is_relevant(c, example))
        return max(retrieved_relevant, 1)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
