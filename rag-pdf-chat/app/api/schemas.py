"""Pydantic request and response schemas for the FastAPI API endpoints."""

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceItem(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    message: str
    filename: str
    total_chunks: int
