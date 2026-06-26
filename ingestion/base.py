"""Abstract interfaces for the ingestion layer.

Defines the three contracts that turn raw sources into indexable vectors:
:class:`Loader` (source -> :class:`Document`), :class:`Chunker`
(:class:`Document` -> chunks), and :class:`Embedder` (text -> vectors).

Concrete implementations land in Sprint 2. These ABCs exist now so that the
rest of the system can be wired and tested against the contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models.domain import Chunk, Document


class Loader(ABC):
    """Loads a source artifact into a normalized :class:`Document`."""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """Load and normalize the artifact at ``path``.

        Parameters
        ----------
        path:
            Filesystem path or URI of the source artifact.

        Returns
        -------
        Document
            The normalized document with extracted text and metadata.
        """
        raise NotImplementedError


class Chunker(ABC):
    """Splits a :class:`Document` into retrievable :class:`Chunk` objects."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into chunks, preserving provenance metadata.

        Implementations MUST propagate ``document_id`` and set
        ``chunk_strategy`` and ``token_count`` on each chunk.
        """
        raise NotImplementedError


class Embedder(ABC):
    """Encodes text into dense embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Parameters
        ----------
        texts:
            Texts to encode. Batching is the caller's responsibility for very
            large inputs; implementations SHOULD honor a configured batch size.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, in the same order.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced vectors (needed to size the index)."""
        raise NotImplementedError
