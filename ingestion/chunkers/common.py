"""Shared helpers for chunkers.

Centralizes chunk construction so every strategy stamps the same provenance and
access metadata consistently: document id, page number, section title, chosen
strategy, token count, and the document-level ACL. This is what makes a chunk
citable and access-controllable downstream.
"""

from __future__ import annotations

import re

from ingestion.segment import Segment
from models.domain import Chunk, ChunkStrategy
from utils.tokenizer import Tokenizer

# Sentence splitter: split after ., !, ? (optionally followed by quotes/brackets)
# when followed by whitespace. Good enough for chunking; not a linguistic parser.
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into trimmed, non-empty sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def make_chunk(
    *,
    text: str,
    document_id: str,
    strategy: ChunkStrategy,
    tokenizer: Tokenizer,
    acl: list[str],
    segment: Segment,
) -> Chunk:
    """Build a :class:`Chunk` with provenance and ACL propagated.

    Parameters
    ----------
    text:
        The chunk text.
    document_id:
        Parent document id.
    strategy:
        The chunking strategy that produced this chunk.
    tokenizer:
        Tokenizer used to compute ``token_count``.
    acl:
        Document-level access-control identifiers, copied onto the chunk.
    segment:
        The source segment, supplying ``page_number`` and ``section_title``.
    """
    return Chunk(
        document_id=document_id,
        text=text,
        page_number=segment.page_number,
        section_title=segment.section_title,
        chunk_strategy=strategy,
        token_count=tokenizer.count(text),
        acl=list(acl),
    )
