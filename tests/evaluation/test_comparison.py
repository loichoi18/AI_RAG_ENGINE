"""Tests for the retrieval/chunking comparison framework (offline)."""

from __future__ import annotations

from evaluation.comparison import compare_retrieval
from evaluation.dataset import GoldenExample
from evaluation.offline import HashingEmbedder
from models.domain import Document
from tests.fakes import FakeVectorStore
from utils.tokenizer import Tokenizer


class _WordTokenizer:
    """Whitespace tokenizer satisfying the Tokenizer protocol."""

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in text.split():
            if w not in self._index:
                self._index[w] = len(self._vocab)
                self._vocab.append(w)
            ids.append(self._index[w])
        return ids

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._vocab[t] for t in tokens)

    def count(self, text: str) -> int:
        return len(text.split())


def _documents() -> list[Document]:
    return [
        Document(
            document_id="deploy",
            source_path="corpus://deploy",
            content="Deployment uses docker compose to roll out services to production.",
        ),
        Document(
            document_id="vpn",
            source_path="corpus://vpn",
            content="VPN setup requires installing the openvpn client and importing a profile.",
        ),
    ]


def _golden() -> list[GoldenExample]:
    return [
        GoldenExample(id="q1", query="docker compose deploy", document_id="deploy",
                      answer_spans=["docker"], answer="docker compose"),
        GoldenExample(id="q2", query="openvpn client setup", document_id="vpn",
                      answer_spans=["openvpn"], answer="openvpn client"),
    ]


def test_compare_retrieval_produces_metrics_per_strategy() -> None:
    tokenizer: Tokenizer = _WordTokenizer()
    table = compare_retrieval(
        _documents(),
        _golden(),
        embedder=HashingEmbedder(dim=128),
        tokenizer=tokenizer,
        make_store=lambda _dim: FakeVectorStore(),
        k=3,
        chunk_size=40,
        chunk_overlap=5,
    )

    assert set(table) == {"dense", "sparse", "hybrid", "multi_query"}
    for metrics in table.values():
        assert "recall@3" in metrics
        assert 0.0 <= metrics["recall@3"] <= 1.0
    # Hybrid should retrieve the relevant doc for these lexically-clear queries.
    assert table["hybrid"]["hit_rate@3"] == 1.0
