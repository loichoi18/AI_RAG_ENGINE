"""Comparison frameworks for chunking and retrieval strategies.

Both follow the same recipe: build an indexed retriever for each variant, run the
:class:`RetrievalEvaluator` over the golden set, and collect a metrics row per
variant. Components (embedder, tokenizer, vector store) are injected, so the
comparisons run offline with deterministic fakes in CI and with real bge + Qdrant
in production — without changing this code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from evaluation.dataset import GoldenExample
from evaluation.retrieval_evaluator import RetrievalEvaluator
from ingestion.base import Chunker, Embedder
from ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker
from models.domain import Chunk, Document
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.multi_query import MultiQueryRetriever
from retrieval.sparse import BM25Retriever
from services.query_rewriting import HeuristicQueryRewriter
from storage.base import VectorStore
from utils.tokenizer import Tokenizer

# A factory that returns a fresh, ready-to-upsert vector store for a dimension.
StoreFactory = Callable[[int], VectorStore]

MetricRow = dict[str, float]
ComparisonTable = dict[str, MetricRow]


def build_chunks(documents: Sequence[Document], chunker: Chunker) -> list[Chunk]:
    """Chunk every document with ``chunker``."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    return chunks


def _dense_retriever(
    chunks: Sequence[Chunk], embedder: Embedder, make_store: StoreFactory
) -> DenseRetriever:
    store = make_store(embedder.dimension)
    store.upsert(list(chunks), embedder.embed([c.text for c in chunks]))
    return DenseRetriever(embedder, store)


def compare_chunking(
    documents: Sequence[Document],
    golden: Sequence[GoldenExample],
    *,
    embedder: Embedder,
    tokenizer: Tokenizer,
    make_store: StoreFactory,
    k: int = 5,
    chunk_size: int = 35,
    chunk_overlap: int = 8,
) -> ComparisonTable:
    """Compare fixed / recursive / semantic chunking on retrieval metrics.

    Retriever is held fixed (hybrid) so differences are attributable to chunking.
    """
    strategies: dict[str, Chunker] = {
        "fixed": FixedSizeChunker(tokenizer, chunk_size, chunk_overlap),
        "recursive": RecursiveChunker(tokenizer, chunk_size, chunk_overlap),
        "semantic": SemanticChunker(embedder, tokenizer, max_tokens=chunk_size),
    }
    table: ComparisonTable = {}
    for name, chunker in strategies.items():
        chunks = build_chunks(documents, chunker)
        dense = _dense_retriever(chunks, embedder, make_store)
        sparse = BM25Retriever(chunks)
        retriever = HybridRetriever(dense, sparse, candidate_k=k * 4)
        table[name] = RetrievalEvaluator(retriever, k=k).evaluate(golden)
    return table


def compare_retrieval(
    documents: Sequence[Document],
    golden: Sequence[GoldenExample],
    *,
    embedder: Embedder,
    tokenizer: Tokenizer,
    make_store: StoreFactory,
    k: int = 5,
    chunk_size: int = 35,
    chunk_overlap: int = 8,
) -> ComparisonTable:
    """Compare dense / sparse / hybrid / multi-query on retrieval metrics.

    Chunking is held fixed (recursive) so differences are attributable to the
    retriever.
    """
    chunker = RecursiveChunker(tokenizer, chunk_size, chunk_overlap)
    chunks = build_chunks(documents, chunker)

    dense = _dense_retriever(chunks, embedder, make_store)
    sparse = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, sparse, candidate_k=k * 4)
    multi = MultiQueryRetriever(
        hybrid, HeuristicQueryRewriter(), max_queries=3, per_query_k=k * 4
    )

    retrievers = {
        "dense": dense,
        "sparse": sparse,
        "hybrid": hybrid,
        "multi_query": multi,
    }
    return {
        name: RetrievalEvaluator(retriever, k=k).evaluate(golden)
        for name, retriever in retrievers.items()
    }
