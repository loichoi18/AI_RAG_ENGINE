"""Recursive, structure-aware chunking.

Greedily packs natural text units into chunks up to a token budget, splitting on
a hierarchy of separators (paragraphs -> lines -> sentences -> words) only as
far down as needed. This respects document structure far better than fixed-size
windows while still honoring the token budget. A token overlap is re-introduced
between consecutive chunks to preserve cross-boundary context.

Trade-offs
----------
* Pro: boundaries fall on natural breaks, improving retrieval coherence.
* Con: chunk sizes vary; a pathological input with no separators degrades to a
  hard token split (handled explicitly).
"""

from __future__ import annotations

from ingestion.base import Chunker
from ingestion.chunkers.common import make_chunk
from ingestion.segment import Segment, get_acl, get_segments
from models.domain import Chunk, ChunkStrategy, Document
from utils.tokenizer import Tokenizer

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker(Chunker):
    """Hierarchical separator-based chunker with token-budgeted packing."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or _DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into structure-aware, token-budgeted chunks."""
        acl = get_acl(document)
        chunks: list[Chunk] = []
        for segment in get_segments(document):
            for text in self._split_segment(segment.text):
                chunks.append(
                    make_chunk(
                        text=text,
                        document_id=document.document_id,
                        strategy=ChunkStrategy.RECURSIVE,
                        tokenizer=self._tokenizer,
                        acl=acl,
                        segment=segment,
                    )
                )
        return chunks

    def _split_segment(self, text: str) -> list[str]:
        """Produce token-budgeted chunk texts for one segment, with overlap."""
        pieces = self._recursive_split(text, self._separators)
        return self._pack(pieces)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Break ``text`` into atomic units no larger than the token budget."""
        text = text.strip()
        if not text:
            return []
        if self._tokenizer.count(text) <= self._chunk_size:
            return [text]
        if not separators:
            # No separators left: hard split on tokens.
            return self._hard_token_split(text)

        sep, *rest = separators
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) == 1:
            # Separator absent; recurse with the next finer separator.
            return self._recursive_split(text, rest)

        units: list[str] = []
        for part in parts:
            if self._tokenizer.count(part) <= self._chunk_size:
                units.append(part.strip())
            else:
                units.extend(self._recursive_split(part, rest))
        return units

    def _hard_token_split(self, text: str) -> list[str]:
        """Last-resort fixed token split for separator-free text."""
        token_ids = self._tokenizer.encode(text)
        out: list[str] = []
        for start in range(0, len(token_ids), self._chunk_size):
            piece = self._tokenizer.decode(token_ids[start : start + self._chunk_size]).strip()
            if piece:
                out.append(piece)
        return out

    def _pack(self, units: list[str]) -> list[str]:
        """Greedily merge atomic units into chunks within the token budget."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._tokenizer.count(unit)
            if current and current_tokens + unit_tokens > self._chunk_size:
                chunks.append(" ".join(current).strip())
                current, current_tokens = self._carry_overlap(current)
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append(" ".join(current).strip())
        return [c for c in chunks if c]

    def _carry_overlap(self, units: list[str]) -> tuple[list[str], int]:
        """Seed the next chunk with trailing units up to the overlap budget."""
        if self._chunk_overlap <= 0:
            return [], 0
        carried: list[str] = []
        tokens = 0
        for unit in reversed(units):
            t = self._tokenizer.count(unit)
            if tokens + t > self._chunk_overlap:
                break
            carried.insert(0, unit)
            tokens += t
        return carried, tokens
