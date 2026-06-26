"""Unit tests proving the abstract interfaces are independently testable.

Two guarantees are asserted:

1. The ABCs cannot be instantiated directly (abstractness is enforced), so an
   incomplete implementation fails loudly rather than silently misbehaving.
2. A trivial in-memory implementation can satisfy a contract and be exercised
   without any infrastructure — this is what lets later sprints unit-test each
   layer in isolation with fakes.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from generation.base import ConfidenceGate
from models.domain import Chunk, RetrievalResult
from retrieval.base import Retriever


def test_cannot_instantiate_abstract_interface() -> None:
    """Abstract contracts reject direct instantiation."""
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        ConfidenceGate()  # type: ignore[abstract]


def test_fake_retriever_satisfies_contract() -> None:
    """A concrete subclass can implement and be tested without infrastructure."""

    class FakeRetriever(Retriever):
        def retrieve(
            self,
            query: str,
            top_k: int,
            acl: Sequence[str] | None = None,
        ) -> list[RetrievalResult]:
            chunk = Chunk(document_id="d1", text=f"answer to: {query}")
            return [RetrievalResult(chunk=chunk, score=1.0, retriever_name="fake")][:top_k]

    results = FakeRetriever().retrieve("what is rrf?", top_k=1)
    assert len(results) == 1
    assert results[0].retriever_name == "fake"


def test_fake_confidence_gate_satisfies_contract() -> None:
    """The refusal gate contract is implementable with a simple threshold rule."""

    class ThresholdGate(ConfidenceGate):
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold

        def passes(self, results: Sequence[RetrievalResult]) -> bool:
            return bool(results) and results[0].score >= self.threshold

    chunk = Chunk(document_id="d1", text="ctx")
    strong = [RetrievalResult(chunk=chunk, score=0.9, retriever_name="fake")]
    weak = [RetrievalResult(chunk=chunk, score=0.1, retriever_name="fake")]

    gate = ThresholdGate(threshold=0.5)
    assert gate.passes(strong) is True
    assert gate.passes(weak) is False
    assert gate.passes([]) is False
