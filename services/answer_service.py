"""Answer orchestration: retrieve -> rerank -> gate -> generate.

This is the end-to-end RAG answer pipeline for a single query. It enforces the
generation rules at the orchestration level:

* **Gate before LLM** — if reranking yields nothing above the confidence
  threshold, the generator is never called and a refusal is returned.
* **Blended confidence** — final confidence is the normalized top reranker score
  multiplied by citation coverage, so an answer that is well-retrieved *and*
  well-grounded scores high, while ungrounded sentences pull it down.
* **Refusal** — both the gate-fail path and the in-generation sentinel produce a
  refusal carrying the configured message and the best documents found, so the
  user is pointed somewhere useful.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from generation.base import ConfidenceGate, LLMProvider
from generation.citations import citation_coverage
from models.domain import GenerationResult, RetrievalResult
from reranking.base import Reranker
from retrieval.base import Retriever
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnswerResponse:
    """The result of answering a query.

    ``refused`` is true when the system declined to answer (low confidence or
    unsupported). ``sources`` are the reranked chunks considered, useful for
    surfacing "recommended documents" on refusal and for debugging.
    """

    answer: str
    citations: list[str]
    confidence: float
    refused: bool
    token_usage: dict[str, int]
    sources: list[RetrievalResult] = field(default_factory=list)


class AnswerService:
    """Coordinates retrieval, reranking, gating, and grounded generation."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        gate: ConfidenceGate,
        generator: LLMProvider,
        candidate_k: int = 20,
        top_k: int = 5,
        refusal_message: str = "I don't have enough information to answer that confidently.",
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._gate = gate
        self._generator = generator
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._refusal_message = refusal_message

    def answer(self, query: str, acl: Sequence[str] | None = None) -> AnswerResponse:
        """Answer ``query`` under the caller's ``acl``, refusing when unsupported."""
        candidates = self._retriever.retrieve(query, self._candidate_k, acl)
        reranked = self._reranker.rerank(query, candidates, self._top_k)

        if not self._gate.passes(reranked):
            logger.info("answer.refused", reason="gate", query=query)
            return self._refuse(reranked)

        result = self._generator.generate(query, [r.chunk for r in reranked])

        # In-generation refusal (sentinel) yields an empty answer.
        if not result.answer.strip():
            logger.info("answer.refused", reason="sentinel", query=query)
            return self._refuse(reranked)

        confidence = self._confidence(reranked, result)
        logger.info(
            "answer.generated",
            query=query,
            citations=len(result.citations),
            confidence=round(confidence, 3),
        )
        return AnswerResponse(
            answer=result.answer,
            citations=result.citations,
            confidence=confidence,
            refused=False,
            token_usage=result.token_usage,
            sources=list(reranked),
        )

    def _confidence(
        self, reranked: Sequence[RetrievalResult], result: GenerationResult
    ) -> float:
        """Blend top reranker score with citation coverage."""
        top_score = max((r.score for r in reranked), default=0.0)
        coverage = citation_coverage(result.answer, len(reranked))
        return round(top_score * coverage, 4)

    def _refuse(self, reranked: Sequence[RetrievalResult]) -> AnswerResponse:
        top_score = max((r.score for r in reranked), default=0.0)
        return AnswerResponse(
            answer=self._refusal_message,
            citations=[],
            confidence=round(top_score * 0.0, 4),
            refused=True,
            token_usage={},
            sources=list(reranked),
        )
