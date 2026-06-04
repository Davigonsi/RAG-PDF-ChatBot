"""Generator module for producing answers using OpenAI GPT given retrieved context."""

from openai import OpenAI

from config.config import get_settings

_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information provided in the context below. If the answer cannot be found "
    "in the context, say exactly: 'I could not find that information in the "
    "provided document.' Always mention the page number(s) your answer is "
    "based on."
)


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """Generate an answer for the query using retrieved context chunks.

    Builds a formatted context string from the provided chunks, then calls
    the OpenAI Chat Completions API with a system prompt that constrains the
    model to answer only from the supplied context. Returns a dict with keys:
      - 'answer': the generated answer string
      - 'sources': list of unique {'source': ..., 'page': ...} dicts
    """
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(
            f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}\n---"
        )
    context_string = "\n".join(context_parts)

    settings = get_settings()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_string}\n\nQuestion: {query}"},
        ],
    )

    answer_text = response.choices[0].message.content

    seen = set()
    sources = []
    for chunk in context_chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": chunk["source"], "page": chunk["page"]})

    return {"answer": answer_text, "sources": sources}
