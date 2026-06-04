"""Tests for retrieval, generation, and the end-to-end RAG chain."""

from unittest.mock import patch

from app.rag.chain import ask

_FAKE_CHUNKS = [
    {"text": "Claude is made by Anthropic.", "source": "doc.pdf", "page": 1, "score": 0.95}
]

_FAKE_ANSWER = {
    "answer": "Claude is made by Anthropic.",
    "sources": [{"source": "doc.pdf", "page": 1}],
}


def test_ask_returns_answer_and_sources():
    with patch("app.rag.chain.retrieve", return_value=_FAKE_CHUNKS), \
         patch("app.rag.chain.generate_answer", return_value=_FAKE_ANSWER):
        result = ask("Who made Claude?")

    assert result["answer"] == "Claude is made by Anthropic."
    assert "sources" in result


def test_ask_with_no_retrieval_results():
    with patch("app.rag.chain.retrieve", return_value=[]):
        result = ask("What is the meaning of life?")

    assert "No relevant content" in result["answer"]
