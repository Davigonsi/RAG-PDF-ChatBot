"""FastAPI application entry point for the RAG PDF Chat API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("uploads").mkdir(exist_ok=True)
    yield


app = FastAPI(title="RAG PDF Chat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["rag"])


@app.get("/")
async def health_check():
    """Return a simple health check response."""
    return {"status": "ok", "model": "gpt-4o-mini"}
