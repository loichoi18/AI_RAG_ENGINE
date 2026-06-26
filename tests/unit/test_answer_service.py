"""Unit tests for the end-to-end answer service (retrieve→rerank→gate→generate)."""

from __future__ import annotations

from collections.abc import Sequence

from generation.gate import ScoreThresholdGate
from generation.generator import GroundedGenerator
from generation.llm.base import Completion, LLMClient
from models.domain import Chunk, RetrievalResult
from reranking.base import Reranker
from retrieval.base import Retriever
from services.answer_service import AnswerService


class _FakeRetriever(Retriever):
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(
        self, query: str, top_k: int, acl: Sequence[str] | None = None
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=c, score=0.5, retriever_name="hybrid")
            for c in self._chunks
        ][:top_k]


class _FixedScoreReranker(Reranker):
    """Assigns a fixed normalized score to every candidate (controls the gate)."""

    def __init__(self, score: float) -> None:
        self._score = score

    def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=r.chunk, score=self._score, retriever_name="cross_encoder")
            for r in results
        ][:top_k]


class _FakeLLMClient(LLMClient):
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        self.calls += 1
        return Completion(text=self._text, prompt_tokens=8, completion_tokens=4)


def _context() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", document_id="doc1", text="alpha fact"),
        Chunk(chunk_id="b", document_id="doc2", text="beta fact"),
    ]


def _service(rerank_score: float, llm_text: str, client_holder: list[_FakeLLMClient]) -> AnswerService:
    client = _FakeLLMClient(llm_text)
    client_holder.append(client)
    return AnswerService(
        retriever=_FakeRetriever(_context()),
        reranker=_FixedScoreReranker(rerank_score),
        gate=ScoreThresholdGate(threshold=0.3),
        generator=GroundedGenerator(client),
        candidate_k=10,
        top_k=2,
        refusal_message="insufficient context",
    )


def test_answer_with_citations_and_confidence() -> None:
    holder: list[_FakeLLMClient] = []
    service = _service(0.8, "Alpha is true [1]. Beta also [2].", holder)

    response = service.answer("what is alpha?")

    assert response.refused is False
    assert response.citations == ["a", "b"]
    # confidence = top rerank score (0.8) * coverage (1.0, both sentences cited)
    assert response.confidence == 0.8
    assert holder[0].calls == 1


def test_low_confidence_gate_refuses_without_calling_llm() -> None:
    holder: list[_FakeLLMClient] = []
    service = _service(0.1, "should never be produced", holder)

    response = service.answer("obscure question")

    assert response.refused is True
    assert response.answer == "insufficient context"
    assert response.citations == []
    assert holder[0].calls == 0  # LLM was never called


def test_sentinel_refusal_after_gate_passes() -> None:
    holder: list[_FakeLLMClient] = []
    service = _service(0.9, "INSUFFICIENT_CONTEXT", holder)

    response = service.answer("question with weak support in text")

    assert response.refused is True
    assert response.answer == "insufficient context"
    assert holder[0].calls == 1  # gate passed, LLM was called, then refused


def test_partial_coverage_lowers_confidence() -> None:
    holder: list[_FakeLLMClient] = []
    service = _service(0.8, "Alpha is true [1]. Ungrounded claim.", holder)

    response = service.answer("what is alpha?")

    assert response.refused is False
    # coverage = 0.5 -> confidence = 0.8 * 0.5 = 0.4
    assert response.confidence == 0.4
