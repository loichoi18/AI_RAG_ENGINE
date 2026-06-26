"""Intermediate ``Segment`` representation produced by loaders.

A loader turns a source file into a :class:`~models.domain.Document` whose flat
``content`` is convenient for full-text use, but provenance (which page, which
section) is lost in a single string. To preserve it, loaders also attach a list
of :class:`Segment` objects under ``document.metadata[SEGMENTS_KEY]``.

Chunkers consume segments so each emitted chunk can carry an accurate
``page_number`` and ``section_title`` for citation. When a document has no
segments (or was constructed by hand), chunkers fall back to treating the whole
``content`` as one segment.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from models.domain import Document

SEGMENTS_KEY = "segments"
ACL_KEY = "acl"


class Segment(BaseModel):
    """A contiguous span of source text with optional provenance."""

    model_config = ConfigDict(extra="forbid")

    text: str
    page_number: int | None = None
    section_title: str | None = None


def attach_segments(document: Document, segments: Sequence[Segment]) -> None:
    """Store ``segments`` on the document's metadata in place."""
    document.metadata[SEGMENTS_KEY] = list(segments)


def get_segments(document: Document) -> list[Segment]:
    """Return the document's segments, or a single whole-content fallback.

    Accepts segments stored either as :class:`Segment` instances or as plain
    dicts (e.g. after JSON round-tripping).
    """
    raw = document.metadata.get(SEGMENTS_KEY)
    if not raw:
        return [Segment(text=document.content)]
    return [s if isinstance(s, Segment) else Segment(**s) for s in raw]


def get_acl(document: Document) -> list[str]:
    """Return the document-level ACL from metadata (empty = unrestricted)."""
    acl = document.metadata.get(ACL_KEY, [])
    return list(acl) if acl else []
