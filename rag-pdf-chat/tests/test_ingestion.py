"""Tests for PDF loading and text chunking in the ingestion pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_loader import load_pdf


def _make_fake_page(text: str) -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    return page


def test_load_pdf_returns_pages():
    fake_pages = [_make_fake_page("Hello world page content") for _ in range(2)]

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.__getitem__ = MagicMock(side_effect=lambda i: fake_pages[i])

    with patch("app.ingestion.pdf_loader.fitz.open", return_value=mock_doc):
        result = load_pdf("fake.pdf")

    assert len(result) == 2
    for item in result:
        assert "page" in item
        assert "text" in item
        assert "source" in item


def test_chunk_documents_splits_text():
    pages = [{"page": 1, "text": "word " * 300, "source": "fake.pdf"}]
    chunks = chunk_documents(pages)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert "page" in chunk
        assert "chunk_index" in chunk


def test_empty_page_is_skipped():
    fake_page = _make_fake_page("   ")

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_doc.__getitem__ = MagicMock(return_value=fake_page)

    with patch("app.ingestion.pdf_loader.fitz.open", return_value=mock_doc):
        result = load_pdf("fake.pdf")

    assert result == []
