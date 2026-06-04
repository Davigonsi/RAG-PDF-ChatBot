"""PDF loader module using PyMuPDF (fitz) to extract text from PDF files."""

import fitz


def load_pdf(file_path: str) -> list[dict]:
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF file '{file_path}': {e}") from e

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            continue
        pages.append({
            "page": page_num + 1,
            "text": text,
            "source": file_path,
        })

    doc.close()
    return pages


if __name__ == "__main__":
    from app.ingestion.chunker import chunk_documents

    path = "uploads/test.pdf"
    pages = load_pdf(path)
    print(f"Pages loaded: {len(pages)}")

    chunks = chunk_documents(pages)
    print(f"Total chunks: {len(chunks)}")
