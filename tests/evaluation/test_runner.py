"""Tests for the evaluation runner and report writing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evaluation.dataset import GoldenExample
from evaluation.runner import EvaluationRunner
from models.domain import Chunk, RetrievalResult
from retrieval.base import Retriever
from services.answer_service import AnswerResponse


class _FakeRetriever(Retriever):
    """Returns one relevant chunk per query, keyed by document id."""

    def __init__(self, mapping: dict[str, tuple[str, str]]) -> None:
        self._mapping = mapping

    def retrieve(
        self, query: str, top_k: int, acl: Sequence[str] | None = None
    ) -> list[RetrievalResult]:
        doc_id, text = self._mapping[query]
        chunk = Chunk(chunk_id=f"{doc_id}-0", document_id=doc_id, text=text)
        return [RetrievalResult(chunk=chunk, score=0.9, retriever_name="fake")]


class _FakeAnswerService:
    """Minimal stand-in for AnswerService with a grounded, cited answer."""

    def __init__(self, mapping: dict[str, tuple[str, str]]) -> None:
        self._mapping = mapping

    def answer(self, query: str, acl: Sequence[str] | None = None) -> AnswerResponse:
        doc_id, text = self._mapping[query]
        chunk = Chunk(chunk_id=f"{doc_id}-0", document_id=doc_id, text=text)
        source = RetrievalResult(chunk=chunk, score=0.9, retriever_name="fake")
        return AnswerResponse(
            answer=f"{text} [1].",
            citations=[chunk.chunk_id],
            confidence=0.8,
            refused=False,
            token_usage={"prompt": 5, "completion": 3},
            sources=[source],
        )


def _dataset() -> list[GoldenExample]:
    return [
        GoldenExample(
            id="q1", query="how to deploy?", document_id="deploy",
            answer_spans=["docker"], answer="deploy with docker",
        ),
        GoldenExample(
            id="q2", query="vpn setup?", document_id="vpn",
            answer_spans=["openvpn"], answer="use openvpn",
        ),
    ]


def _mapping() -> dict[str, tuple[str, str]]:
    return {
        "how to deploy?": ("deploy", "deploy with docker compose"),
        "vpn setup?": ("vpn", "configure openvpn client"),
    }


def test_runner_retrieval_only() -> None:
    runner = EvaluationRunner(_FakeRetriever(_mapping()), k=3, name="retr_only")
    report = runner.run(_dataset())

    assert report.dataset_size == 2
    assert report.retrieval["recall@3"] == 1.0  # the relevant doc is retrieved each time
    assert report.retrieval["hit_rate@3"] == 1.0
    assert report.generation == {}
    assert len(report.per_question) == 2


def test_runner_with_generation_and_reports(tmp_path: Path) -> None:
    runner = EvaluationRunner(
        _FakeRetriever(_mapping()),
        answer_service=_FakeAnswerService(_mapping()),  # type: ignore[arg-type]
        k=3,
        name="full",
    )
    report = runner.run(_dataset(), output_dir=tmp_path, formats=("json", "md", "csv"))

    assert "faithfulness" in report.generation
    assert report.generation["refusal_rate"] == 0.0
    # Reports written to disk.
    assert (tmp_path / "full.json").exists()
    assert (tmp_path / "full.md").exists()
    assert (tmp_path / "full.csv").exists()
