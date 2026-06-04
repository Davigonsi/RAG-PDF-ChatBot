"""Chain module that combines retrieval and generation into a full RAG pipeline."""

from app.rag.retriever import retrieve
from app.rag.generator import generate_answer


def ask(query: str, top_k: int = 5) -> dict:
    """Run the full RAG pipeline for a user query.

    Retrieves the top_k most relevant chunks from the vector store, then
    passes them to the generator to produce a grounded answer. If no
    relevant chunks are found, returns a canned 'no content' response.
    Returns a dict with keys 'answer' and 'sources'.
    """
    chunks = retrieve(query, top_k)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded document.",
            "sources": [],
        }

    return generate_answer(query, chunks)
