"""Plain-text loader."""

from __future__ import annotations

from pathlib import Path

from ingestion.base import Loader
from ingestion.segment import Segment, attach_segments
from models.domain import Document


class TextLoader(Loader):
    """Loads ``.txt`` files as a single-segment document."""

    def load(self, path: str | Path) -> Document:
        """Read ``path`` as UTF-8 text into a :class:`Document`."""
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        document = Document(
            source_path=str(p),
            content=content,
            metadata={"mime_type": "text/plain", "loader": "text"},
        )
        attach_segments(document, [Segment(text=content)])
        return document
