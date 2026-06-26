"""Markdown loader.

Splits a Markdown document into one :class:`Segment` per section, keyed by ATX
headings (``#``..``######``). Each segment records its nearest heading as
``section_title`` so chunks can cite the section. The flat ``content`` keeps the
raw Markdown for full-text use.
"""

from __future__ import annotations

import re
from pathlib import Path

from ingestion.base import Loader
from ingestion.segment import Segment, attach_segments
from models.domain import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownLoader(Loader):
    """Loads ``.md`` / ``.markdown`` files, one segment per heading section."""

    def load(self, path: str | Path) -> Document:
        """Parse ``path`` into heading-delimited segments."""
        p = Path(path)
        content = p.read_text(encoding="utf-8")

        segments: list[Segment] = []
        current_title: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                segments.append(Segment(text=text, section_title=current_title))
            buffer.clear()

        for line in content.splitlines():
            match = _HEADING.match(line)
            if match:
                flush()
                current_title = match.group(2).strip()
            else:
                buffer.append(line)
        flush()

        if not segments:
            segments = [Segment(text=content.strip())]

        document = Document(
            source_path=str(p),
            content=content,
            metadata={"mime_type": "text/markdown", "loader": "markdown"},
        )
        attach_segments(document, segments)
        return document
