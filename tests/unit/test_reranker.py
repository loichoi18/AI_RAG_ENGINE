"""Unit tests for the cross-encoder reranker (model patched with a fake)."""

from __future__ import annotations

import sys
import types

import pytest

from models.domain import Chunk, RetrievalResult
from reranking.cross_encoder import CrossEncoderReranker, _sigmoid


def _result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=chunk_id, document_id="d", text=text),
        score=score,
        retriever_name="hybrid",
    )


@pytest.fixture
def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch sentence_transformers.CrossEncoder with a keyword-scoring fake."""

    class _FakeCE:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            pass

        def predict(self, pairs: list[list[str]]):
            # Score by presence of "match" in the candidate text (as a logit).
            return [3.0 if "match" in text else -3.0 for _query, text in pairs]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _FakeCE  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_sigmoid_bounds() -> None:
    assert 0.0 < _sigmoid(0.0) < 1.0
    assert _sigmoid(10.0) > 0.99
    assert _sigmoid(-10.0) < 0.01


def test_reranker_reorders_by_joint_relevance(fake_cross_encoder: None) -> None:
    reranker = CrossEncoderReranker(model_name="fake")
    # First-stage order puts the irrelevant one first; reranker must fix that.
    candidates = [
        _result("a", "irrelevant text", 0.9),
        _result("b", "this is a match", 0.1),
    ]
    out = reranker.rerank("q", candidates, top_k=2)

    assert out[0].chunk.chunk_id == "b"
    assert out[0].score > 0.9  # sigmoid(3.0)
    assert all(r.retriever_name == "cross_encoder" for r in out)


def test_reranker_truncates_to_top_k(fake_cross_encoder: None) -> None:
    candidates = [_result(str(i), "match", 0.5) for i in range(5)]
    assert len(CrossEncoderReranker(model_name="fake").rerank("q", candidates, top_k=2)) == 2


def test_reranker_handles_empty(fake_cross_encoder: None) -> None:
    assert CrossEncoderReranker(model_name="fake").rerank("q", [], top_k=3) == []
