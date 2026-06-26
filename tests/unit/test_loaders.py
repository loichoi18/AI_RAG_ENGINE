"""Unit tests for document loaders.

Strategy: write tiny real files for text/markdown/html and assert that content,
segment provenance (page/section), and metadata are extracted correctly. The
PDF loader is tested with a patched ``PdfReader`` so the page->segment mapping
is verified without shipping a binary fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ingestion.loaders.pdf_loader as pdf_module
from ingestion.loaders import (
    HtmlLoader,
    LoaderRegistry,
    MarkdownLoader,
    PdfLoader,
    TextLoader,
)
from ingestion.segment import get_segments


def test_text_loader(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("hello world", encoding="utf-8")

    doc = TextLoader().load(f)

    assert doc.content == "hello world"
    assert doc.metadata["loader"] == "text"
    segments = get_segments(doc)
    assert len(segments) == 1
    assert segments[0].text == "hello world"


def test_markdown_loader_segments_by_heading(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(
        "# Intro\nWelcome text.\n\n## Setup\nInstall steps here.\n",
        encoding="utf-8",
    )

    doc = MarkdownLoader().load(f)
    segments = get_segments(doc)

    titles = [s.section_title for s in segments]
    assert "Intro" in titles
    assert "Setup" in titles
    setup = next(s for s in segments if s.section_title == "Setup")
    assert "Install steps" in setup.text


def test_html_loader_strips_noise_and_segments(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><title>Docs</title></head><body>"
        "<script>evil()</script>"
        "<h1>Overview</h1><p>Body paragraph.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    doc = HtmlLoader().load(f)

    assert "evil" not in doc.content
    assert doc.metadata["title"] == "Docs"
    segments = get_segments(doc)
    assert any(s.section_title == "Overview" and "Body paragraph" in s.text for s in segments)


def test_pdf_loader_maps_pages_to_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [_FakePage("Page one text."), _FakePage("Page two text.")]

    monkeypatch.setattr(pdf_module, "PdfReader", _FakeReader)

    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    doc = PdfLoader().load(f)
    segments = get_segments(doc)

    assert [s.page_number for s in segments] == [1, 2]
    assert segments[0].text == "Page one text."
    assert doc.metadata["loader"] == "pdf"


def test_registry_resolves_and_rejects(tmp_path: Path) -> None:
    registry = LoaderRegistry()
    assert isinstance(registry.get("a.md"), MarkdownLoader)
    assert isinstance(registry.get("b.PDF"), PdfLoader)
    with pytest.raises(ValueError):
        registry.get("archive.zip")
