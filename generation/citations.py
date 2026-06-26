"""Citation parsing, validation, and coverage.

Generated answers cite context blocks as ``[n]``. This module extracts those
markers, maps each to the corresponding chunk, drops citations that reference a
non-existent block (a hallucination signal), and computes **citation
coverage** — the fraction of answer sentences carrying at least one valid
citation. Coverage feeds the confidence score: ungrounded sentences lower it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from models.domain import Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Citation(BaseModel):
    """A validated mapping from an answer marker to a source chunk."""

    index: int
    chunk_id: str
    document_id: str
    section_title: str | None = None
    page_number: int | None = None


def extract_citation_indices(answer: str) -> list[int]:
    """Return all bracketed citation numbers in the order they appear."""
    return [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]


def parse_citations(answer: str, context: Sequence[Chunk]) -> list[Citation]:
    """Map valid ``[n]`` markers in ``answer`` to context chunks.

    Indices outside ``1..len(context)`` are ignored (hallucinated citations).
    Returns distinct citations in first-appearance order.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    for index in extract_citation_indices(answer):
        if index in seen or not (1 <= index <= len(context)):
            continue
        seen.add(index)
        chunk = context[index - 1]
        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
            )
        )
    return citations


def citation_coverage(answer: str, context_size: int) -> float:
    """Fraction of answer sentences containing at least one valid citation.

    A sentence is "covered" if it cites a block in ``1..context_size``. Returns
    0.0 for an empty answer; 1.0 when every sentence is grounded.
    """
    sentences = [s for s in _SENTENCE_RE.split(answer.strip()) if s.strip()]
    if not sentences:
        return 0.0
    covered = 0
    for sentence in sentences:
        indices = extract_citation_indices(sentence)
        if any(1 <= i <= context_size for i in indices):
            covered += 1
    return covered / len(sentences)
