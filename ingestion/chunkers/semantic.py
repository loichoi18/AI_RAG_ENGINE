"""Semantic chunking via embedding-similarity breakpoints.

Embeds each sentence, then walks adjacent sentences computing cosine similarity.
A *breakpoint* is placed where the similarity between consecutive sentences
falls below a configured percentile of all adjacent similarities — i.e. where
the topic shifts. Sentences between breakpoints form a chunk, subject to a
maximum token budget so chunks never exceed the embedder's context.

Trade-offs
----------
* Pro: boundaries track meaning, often yielding the most coherent chunks and the
  best retrieval quality.
* Con: cost. It embeds every sentence at ingest time, so it is the slowest
  strategy and depends on the :class:`Embedder`. This is why it is opt-in and
  benchmarked against cheaper strategies rather than assumed best.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ingestion.base import Chunker, Embedder
from ingestion.chunkers.common import make_chunk, split_sentences
from ingestion.segment import get_acl, get_segments
from models.domain import Chunk, ChunkStrategy, Document
from utils.tokenizer import Tokenizer


class SemanticChunker(Chunker):
    """Topic-aware chunker using sentence-embedding similarity breakpoints."""

    def __init__(
        self,
        embedder: Embedder,
        tokenizer: Tokenizer,
        max_tokens: int = 512,
        threshold_percentile: float = 95.0,
    ) -> None:
        if not 0.0 <= threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be in [0, 100]")
        self._embedder = embedder
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._threshold_percentile = threshold_percentile

    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into semantically coherent chunks."""
        acl = get_acl(document)
        chunks: list[Chunk] = []
        for segment in get_segments(document):
            for text in self._split_segment(segment.text):
                chunks.append(
                    make_chunk(
                        text=text,
                        document_id=document.document_id,
                        strategy=ChunkStrategy.SEMANTIC,
                        tokenizer=self._tokenizer,
                        acl=acl,
                        segment=segment,
                    )
                )
        return chunks

    def _split_segment(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        vectors = self._embedder.embed(sentences)
        similarities = [
            _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]
        threshold = _percentile(similarities, 100.0 - self._threshold_percentile)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            prospective = " ".join([*current, sentences[i]])
            over_budget = self._tokenizer.count(prospective) > self._max_tokens
            topic_shift = similarities[i - 1] < threshold
            if current and (over_budget or topic_shift):
                chunks.append(" ".join(current).strip())
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            chunks.append(" ".join(current).strip())
        return [c for c in chunks if c]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0.0 if either is a zero vector)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolation percentile of ``values`` (empty -> 0.0)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
