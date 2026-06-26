"""HTML loader built on BeautifulSoup.

Strips script/style noise and extracts readable text. Content is segmented by
heading tags (``h1``..``h6``); each segment records its heading as
``section_title``. The document title (``<title>``) is captured in metadata.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ingestion.base import Loader
from ingestion.segment import Segment, attach_segments
from models.domain import Document

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_NOISE_TAGS = ["script", "style", "noscript"]


class HtmlLoader(Loader):
    """Loads ``.html`` / ``.htm`` files, segmented by heading tags."""

    def load(self, path: str | Path) -> Document:
        """Parse ``path`` into readable text segmented by headings."""
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "html.parser")

        for noise in soup(_NOISE_TAGS):
            noise.decompose()

        page_title = soup.title.string.strip() if soup.title and soup.title.string else None
        body = soup.body or soup

        segments: list[Segment] = []
        current_title: str | None = page_title
        buffer: list[str] = []

        def flush() -> None:
            text = " ".join(" ".join(buffer).split()).strip()
            if text:
                segments.append(Segment(text=text, section_title=current_title))
            buffer.clear()

        for element in body.descendants:
            if isinstance(element, Tag) and element.name in _HEADING_TAGS:
                flush()
                current_title = element.get_text(strip=True) or current_title
            elif isinstance(element, str):
                buffer.append(element)
        flush()

        content = "\n\n".join(seg.text for seg in segments)
        if not content.strip():
            raise ValueError(f"No extractable text found in HTML: {p}")

        document = Document(
            source_path=str(p),
            content=content,
            metadata={
                "mime_type": "text/html",
                "loader": "html",
                "title": page_title,
            },
        )
        attach_segments(document, segments)
        return document
