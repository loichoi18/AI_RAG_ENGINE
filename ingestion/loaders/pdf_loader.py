"""PDF loader built on ``pypdf``.

Each PDF page becomes one :class:`Segment` carrying its 1-indexed
``page_number`` so chunks can cite the exact page. ``PdfReader`` is imported at
module load but referenced through a module attribute, which lets tests patch
it with a fake reader and exercise the page->segment mapping without a real PDF.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ingestion.base import Loader
from ingestion.segment import Segment, attach_segments
from models.domain import Document


class PdfLoader(Loader):
    """Loads ``.pdf`` files, one segment per page."""

    def load(self, path: str | Path) -> Document:
        """Extract text per page into a paginated :class:`Document`."""
        p = Path(path)
        reader = PdfReader(str(p))

        segments: list[Segment] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            segments.append(Segment(text=text, page_number=index))

        content = "\n\n".join(seg.text for seg in segments)
        if not content.strip():
            raise ValueError(f"No extractable text found in PDF: {p}")

        document = Document(
            source_path=str(p),
            content=content,
            metadata={
                "mime_type": "application/pdf",
                "loader": "pdf",
                "page_count": len(reader.pages),
            },
        )
        attach_segments(document, segments)
        return document
