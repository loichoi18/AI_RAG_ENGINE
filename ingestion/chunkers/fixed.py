"""Fixed-size token chunking with overlap.

Splits each segment into windows of ``chunk_size`` tokens that slide forward by
``chunk_size - chunk_overlap`` tokens. Overlap preserves context that would
otherwise be severed at a hard boundary.

Trade-offs
----------
* Pro: simple, deterministic, predictable cost; never exceeds the embedder's
  context budget.
* Con: boundaries ignore semantics, so a sentence (or idea) can be split across
  two chunks. Overlap mitigates but does not eliminate this.
"""

from __future__ import annotations

from ingestion.base import Chunker
from ingestion.chunkers.common import make_chunk
from ingestion.segment import get_acl, get_segments
from models.domain import Chunk, ChunkStrategy, Document
from utils.tokenizer import Tokenizer


class FixedSizeChunker(Chunker):
    """Token-window chunker with configurable size and overlap."""

    def __init__(self, tokenizer: Tokenizer, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into fixed-size overlapping token windows."""
        acl = get_acl(document)
        step = self._chunk_size - self._chunk_overlap
        chunks: list[Chunk] = []

        for segment in get_segments(document):
            token_ids = self._tokenizer.encode(segment.text)
            if not token_ids:
                continue
            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + self._chunk_size]
                if not window:
                    continue
                text = self._tokenizer.decode(window).strip()
                if not text:
                    continue
                chunks.append(
                    make_chunk(
                        text=text,
                        document_id=document.document_id,
                        strategy=ChunkStrategy.FIXED,
                        tokenizer=self._tokenizer,
                        acl=acl,
                        segment=segment,
                    )
                )
                if start + self._chunk_size >= len(token_ids):
                    break
        return chunks
