"""Vector store module for upserting and querying embeddings in Pinecone."""

from pinecone import Pinecone

from config.config import get_settings

_index = None


def get_index():
    """Return the Pinecone index singleton, initializing it on first call.

    Uses PINECONE_API_KEY and PINECONE_INDEX_NAME from settings.
    """
    global _index
    if _index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _index


def upsert_chunks(embedded_chunks: list[dict]) -> None:
    """Upsert embedded chunk dicts into the Pinecone index in batches of 100.

    Each vector is identified by a string ID derived from the source path and
    chunk index. Metadata stored per vector: text, source, and page number.
    """
    index = get_index()

    vectors = []
    for chunk in embedded_chunks:
        vec_id = f"{chunk['source']}__chunk_{chunk['chunk_index']}"
        metadata = {
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
        }
        vectors.append((vec_id, chunk["embedding"], metadata))

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Query the Pinecone index for the top-k nearest neighbours.

    Returns a list of metadata dicts from the matched vectors, each augmented
    with a 'score' field containing the similarity score.
    """
    index = get_index()

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    hits = []
    for match in result.matches:
        entry = dict(match.metadata)
        entry["score"] = match.score
        hits.append(entry)

    return hits
