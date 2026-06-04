"""Embeddings package for generating and storing vector embeddings."""

from app.embeddings.embedder import embed_chunks, embed_query
from app.embeddings.vector_store import upsert_chunks, search

__all__ = ["embed_chunks", "embed_query", "upsert_chunks", "search"]
