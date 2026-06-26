"""Loader registry: maps file extensions to the appropriate loader.

Centralizing the extension dispatch (Factory pattern) means callers ingest a
path without knowing the concrete loader, and new formats are added in one
place. Unknown extensions raise a clear error rather than silently mis-parsing.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.base import Loader
from ingestion.loaders.html_loader import HtmlLoader
from ingestion.loaders.markdown_loader import MarkdownLoader
from ingestion.loaders.pdf_loader import PdfLoader
from ingestion.loaders.text_loader import TextLoader
from models.domain import Document


class LoaderRegistry:
    """Resolves a :class:`Loader` for a given file path by extension."""

    def __init__(self) -> None:
        text = TextLoader()
        markdown = MarkdownLoader()
        html = HtmlLoader()
        pdf = PdfLoader()
        self._by_extension: dict[str, Loader] = {
            ".txt": text,
            ".text": text,
            ".md": markdown,
            ".markdown": markdown,
            ".html": html,
            ".htm": html,
            ".pdf": pdf,
        }

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Extensions this registry can load."""
        return frozenset(self._by_extension)

    def get(self, path: str | Path) -> Loader:
        """Return the loader for ``path``.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        suffix = Path(path).suffix.lower()
        try:
            return self._by_extension[suffix]
        except KeyError:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported: {sorted(self.supported_extensions)}"
            ) from None

    def load(self, path: str | Path) -> Document:
        """Convenience: resolve the loader and load ``path`` in one call."""
        return self.get(path).load(path)
