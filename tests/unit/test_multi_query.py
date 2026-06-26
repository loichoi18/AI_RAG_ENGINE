"""Unit tests for the multi-query retriever."""

from __future__ import annotations

from collections.abc import Sequence

from models.domain import Chunk, RetrievalResult
from retrieval.base import Retriever
from retrieval.multi_query import MultiQueryRetriever
from services.query_rewriting import HeuristicQueryRewriter


class _PerVariantRetriever(Retriever):
    """Returns a distinct chunk per query variant, so fusion must union them."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def retrieve(
        self, query: str, top_k: int, acl: Sequence[str] | None = None
    ) -> list[RetrievalResult]:
        chunk_id = self._mapping.get(query)
        if chunk_id is None:
            return []
        chunk = Chunk(chunk_id=chunk_id, document_id="d", text=f"for {query}")
        return [RetrievalResult(chunk=chunk, score=1.0, retriever_name="fake")]


def test_multi_query_unions_variant_results() -> None:
    rewriter = HeuristicQueryRewriter()
    variants = rewriter.expand_query("How do we deploy services?")
    mapping = {variant: f"chunk-{i}" for i, variant in enumerate(variants)}

    retriever = MultiQueryRetriever(
        _PerVariantRetriever(mapping), rewriter, max_queries=len(variants)
    )
    results = retriever.retrieve("How do we deploy services?", top_k=10)

    returned = {r.chunk.chunk_id for r in results}
    assert returned == set(mapping.values())
    assert all(r.retriever_name == "multi_query" for r in results)
