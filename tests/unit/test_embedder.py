"""Unit test for the BGE embedder.

The real ``SentenceTransformer`` is replaced with a fake module so we verify
batching, normalization plumbing, and output shape without downloading weights.
"""

from __future__ import annotations

import sys
import types

import pytest

from ingestion.embedding import BGEEmbedder


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    np = pytest.importorskip("numpy")
    calls: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            calls["model_name"] = model_name
            calls["device"] = device

        def encode(self, texts: list[str], **kwargs: object):
            calls["encode_kwargs"] = kwargs
            return np.array([[float(len(t)), 1.0, 2.0] for t in texts])

        def get_sentence_embedding_dimension(self) -> int:
            return 3

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def test_embed_returns_one_vector_per_text(
    fake_sentence_transformers: dict[str, object],
) -> None:
    embedder = BGEEmbedder(model_name="fake/model", normalize=True, batch_size=8)
    vectors = embedder.embed(["a", "bb", "ccc"])

    assert len(vectors) == 3
    assert all(len(v) == 3 for v in vectors)
    assert vectors[0][0] == 1.0  # len("a")
    # normalize + batch_size are forwarded to the model.
    kwargs = fake_sentence_transformers["encode_kwargs"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 8


def test_embed_empty_returns_empty(fake_sentence_transformers: dict[str, object]) -> None:
    assert BGEEmbedder(model_name="fake/model").embed([]) == []


def test_dimension_reported_from_model(
    fake_sentence_transformers: dict[str, object],
) -> None:
    assert BGEEmbedder(model_name="fake/model").dimension == 3
