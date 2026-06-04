"""Embedder module using OpenAI text-embedding-3-small to generate vector embeddings."""

from openai import OpenAI

from config.config import get_settings

_MODEL = "text-embedding-3-small"


_EMBED_BATCH_SIZE = 50


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed a list of chunk dicts using OpenAI text-embedding-3-small.

    Sends chunks in batches of 100 to stay within the API token-per-request
    limit. Each chunk dict is augmented with an 'embedding' key containing
    the corresponding vector (list of floats). Returns a new list of dicts.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    embedded = []
    for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[i : i + _EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        response = client.embeddings.create(input=texts, model=_MODEL)
        for chunk, embedding_obj in zip(batch, response.data):
            embedded.append({**chunk, "embedding": embedding_obj.embedding})

    return embedded


def embed_query(query: str) -> list[float]:
    """Embed a single query string using OpenAI text-embedding-3-small.

    Returns the embedding vector as a list of floats.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.embeddings.create(input=[query], model=_MODEL)
    return response.data[0].embedding
