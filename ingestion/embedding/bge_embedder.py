"""Dense embedder backed by ``sentence-transformers`` (BAAI/bge-base-en-v1.5).

The model is loaded lazily on first use so that importing this module — and
running unit tests that inject fakes — never triggers a multi-hundred-MB
download. ``bge`` models expect L2-normalized embeddings for cosine similarity,
which is enabled by default via configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ingestion.base import Embedder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer


class BGEEmbedder(Embedder):
    """Sentence-Transformers embedder with lazy model loading and batching."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model: SentenceTransformer | None = None

    def _load(self) -> "SentenceTransformer":
        """Load and cache the underlying model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, returning one vector per input."""
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    @property
    def dimension(self) -> int:
        """Embedding dimensionality reported by the loaded model."""
        dim = self._load().get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover - defensive
            raise RuntimeError("Embedding model did not report a dimension")
        return int(dim)
