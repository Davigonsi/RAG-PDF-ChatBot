"""Ingestion package for loading and chunking PDF documents."""

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents

__all__ = ["load_pdf", "chunk_documents"]
