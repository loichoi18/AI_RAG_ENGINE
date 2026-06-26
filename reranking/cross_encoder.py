"""Cross-encoder reranker (second-stage retrieval).

A cross-encoder reads the (query, chunk) pair *jointly* and outputs a single
relevance score, which is far more precise than the bi-encoder cosine used for
first-stage retrieval — at the cost of running the model once per candidate.
That is exactly why it sits second in the funnel, re-scoring only the ~20
candidates that retrieval already narrowed down.

Raw cross-encoder outputs are unbounded logits. We squash them through a sigmoid
to [0, 1] so the score is interpretable as a confidence and can drive the
refusal gate with a fixed threshold. The model is loaded lazily so tests inject
a fake without downloading weights.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from models.domain import RetrievalResult
from reranking.base import Reranker
from utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic squashing to (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderReranker(Reranker):
    """Re-scores candidates with a sentence-transformers ``CrossEncoder``."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model: CrossEncoder | None = None

    def _load(self) -> "CrossEncoder":
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score ``results`` against ``query`` and return the best ``top_k``."""
        if not results:
            return []
        started = time.perf_counter()

        pairs = [[query, r.chunk.text] for r in results]
        raw_scores = self._load().predict(pairs)

        rescored = [
            RetrievalResult(
                chunk=result.chunk,
                score=_sigmoid(float(score)) if self._normalize else float(score),
                retriever_name=self.name,
            )
            for result, score in zip(results, raw_scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        top = rescored[:top_k]
        logger.info(
            "rerank.cross_encoder",
            candidates=len(results),
            returned=len(top),
            top_score=top[0].score if top else None,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return top
