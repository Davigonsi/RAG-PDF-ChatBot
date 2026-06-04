"""Chunker module using LangChain RecursiveCharacterTextSplitter to split text into chunks."""

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = []
    chunk_index = 0
    for page in pages:
        split_texts = splitter.split_text(page["text"])
        for text in split_texts:
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return chunks
