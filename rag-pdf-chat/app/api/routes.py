"""FastAPI route definitions for PDF upload and chat query endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import AskRequest, AskResponse, UploadResponse
from app.embeddings.embedder import embed_chunks
from app.embeddings.vector_store import upsert_chunks
from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_loader import load_pdf
from app.rag.chain import ask

router = APIRouter()

_UPLOAD_DIR = Path("uploads")
_CHUNK_SIZE = 1024 * 1024  # 1 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile):
    """Accept a PDF file, ingest it through the full pipeline, and store embeddings.

    Validates content type, saves the file to disk, then runs load → chunk →
    embed → upsert. Returns the filename and total number of chunks created.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        save_path = _UPLOAD_DIR / file.filename
        with open(save_path, "wb") as out:
            while chunk := await file.read(_CHUNK_SIZE):
                out.write(chunk)

        pages = load_pdf(str(save_path))
        chunks = chunk_documents(pages)
        embedded = embed_chunks(chunks)
        upsert_chunks(embedded)

        return UploadResponse(
            message="PDF processed successfully",
            filename=file.filename,
            total_chunks=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    """Answer a question using the RAG pipeline over previously uploaded documents.

    Retrieves relevant chunks from the vector store and generates a grounded
    answer via the Anthropic Claude API.
    """
    try:
        result = ask(body.query, body.top_k)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
