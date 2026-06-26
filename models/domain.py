"""Core domain models for the RAG engine.

These Pydantic models are the **data contracts** shared across every layer
(ingestion, retrieval, reranking, generation, evaluation). They intentionally
contain no behavior beyond validation: keeping them pure means any layer can
depend on them without creating import cycles.

Design notes
------------
* ``StrEnum`` is used for closed vocabularies (chunking strategy) so invalid
  values fail fast at construction time rather than deep inside a pipeline.
* ``model_config = ConfigDict(frozen=False, extra="forbid")`` rejects unknown
  fields, surfacing typos early — important when these objects are built from
  external loaders.
* Provenance fields on :class:`Chunk` (``page_number``, ``section_title``,
  ``chunk_strategy``) exist so generated answers can cite an exact location,
  which is a hard requirement for an internal knowledge tool.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkStrategy(StrEnum):
    """Supported chunking strategies.

    Stored on each chunk so evaluation can compare strategies on identical
    corpora and queries.
    """

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class Document(BaseModel):
    """A normalized source document prior to chunking.

    Attributes
    ----------
    document_id:
        Stable identifier. Defaults to a UUID4 hex string so ingestion can run
        without the caller pre-assigning ids, but callers SHOULD supply a
        deterministic id (e.g. a content hash or source URI) to make
        re-ingestion idempotent.
    source_path:
        Origin of the document (file path, URL, or connector URI).
    content:
        Full normalized text content.
    metadata:
        Free-form source metadata (author, mime type, tags, ...).
    source_updated_at:
        Last-modified timestamp at the source, used for freshness / re-indexing.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(default_factory=lambda: uuid4().hex)
    source_path: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_updated_at: datetime | None = None

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        """Reject whitespace-only documents that would yield empty chunks."""
        if not value.strip():
            raise ValueError("Document.content must not be empty or whitespace-only")
        return value


class Chunk(BaseModel):
    """A retrievable unit of text with full provenance and access metadata.

    The provenance fields (``page_number``, ``section_title``) enable citation
    back to a precise location. ``acl`` is the access-control list: the set of
    group/user identifiers permitted to retrieve this chunk. An empty list means
    "no explicit restriction" under the current permissive default policy; this
    is enforced as a query-time filter, never as post-retrieval redaction.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    text: str
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    token_count: int = Field(default=0, ge=0)
    acl: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        """A chunk with no text is never useful; reject it at construction."""
        if not value.strip():
            raise ValueError("Chunk.text must not be empty or whitespace-only")
        return value


class RetrievalResult(BaseModel):
    """A chunk returned by a retriever, with its score and origin.

    ``retriever_name`` records which retriever (e.g. ``"dense"``, ``"sparse"``,
    ``"hybrid"``) produced the result. This is essential for debugging fusion
    and for evaluation that breaks metrics down per retriever.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float
    retriever_name: str


class GenerationResult(BaseModel):
    """The output of the generation layer.

    ``citations`` are chunk ids referenced by the answer, preserving
    traceability from answer back to source. ``confidence`` is a normalized
    [0, 1] signal (e.g. derived from the top reranker score) used by the
    refusal gate. ``token_usage`` records prompt/completion tokens for
    observability and cost tracking.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    token_usage: dict[str, int] = Field(default_factory=dict)
