"""Retriever module for fetching relevant document chunks from the vector store."""

from app.embeddings.embedder import embed_query
from app.embeddings.vector_store import search


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant document chunks for a given query.

    Embeds the query string using OpenAI, then searches the Pinecone index
    for the top_k nearest vectors. Returns a list of chunk metadata dicts,
    each containing 'text', 'source', 'page', and 'score'.
    """
    query_vector = embed_query(query)
    results = search(query_vector, top_k)
    return results
