"""Document loaders for supported source formats."""

from ingestion.loaders.html_loader import HtmlLoader
from ingestion.loaders.markdown_loader import MarkdownLoader
from ingestion.loaders.pdf_loader import PdfLoader
from ingestion.loaders.registry import LoaderRegistry
from ingestion.loaders.text_loader import TextLoader

__all__ = [
    "HtmlLoader",
    "LoaderRegistry",
    "MarkdownLoader",
    "PdfLoader",
    "TextLoader",
]
