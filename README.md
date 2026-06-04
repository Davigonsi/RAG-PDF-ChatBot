![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-412991?logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-00C5A8)

# RAG PDF Chat

Upload one or more PDF documents and ask questions across all of them in a single chat interface. Answers are grounded strictly in your documents — every response includes the page number(s) it came from.

---

## What is it and why use it?

Traditional search finds keywords. **RAG PDF Chat understands meaning.**

- Ask questions in plain English — no need to know which page or section to look in
- Upload multiple PDFs and ask cross-document questions in the same conversation
- Every answer cites the exact page(s) it came from so you can verify
- Nothing leaves your session — your documents are only used to answer your questions
- Works with any PDF: textbooks, research papers, contracts, reports, manuals

**Powered by:**
- **OpenAI `gpt-4o-mini`** for fast, accurate answer generation
- **OpenAI `text-embedding-3-small`** for semantic search embeddings
- **Pinecone** serverless vector database for similarity search
- **FastAPI** backend + **Streamlit** frontend

---

## How it works

```
PDF Upload
   │
   ▼
PyMuPDF extracts text page by page
   │
   ▼
LangChain splits text into 500-char overlapping chunks
   │
   ▼
OpenAI embeds each chunk → 1536-dim vectors (batched, 50 at a time)
   │
   ▼
Pinecone stores all vectors (dense, cosine similarity index)
   │
   ▼
User asks a question
   │
   ▼
Question is embedded → top 5 matching chunks retrieved from Pinecone
   │
   ▼
GPT-4o-mini reads the chunks and generates a grounded answer with page citations
```

---

## Tech stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| PDF parsing | PyMuPDF | 1.24.10 | Extract text page by page |
| Chunking | LangChain | 0.2.16 | Split text into 500-char overlapping segments |
| Embeddings | OpenAI text-embedding-3-small | — | Convert text to 1536-dim vectors |
| Vector DB | Pinecone serverless | 6.0.2 | Store and search dense embeddings |
| LLM | OpenAI gpt-4o-mini | — | Generate grounded answers from retrieved context |
| Backend | FastAPI | 0.115.0 | REST API — `/api/upload` and `/api/ask` |
| Frontend | Streamlit | 1.38.0 | Wide-layout chat UI with multi-PDF upload |
| Config | pydantic-settings | 2.5.2 | Secure environment variable loading |

---

## Getting started

### Prerequisites

- Python 3.11+
- OpenAI API key → https://platform.openai.com
- Pinecone account → https://pinecone.io

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If `uvicorn` or `streamlit` are not recognised as commands, use
> `python -m uvicorn` and `python -m streamlit` instead (see step 6 & 7).

### 4. Configure environment variables

Create a `.env` file in the project root (do **not** commit this file):

```
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=ragpdfbot
PINECONE_ENVIRONMENT=us-east-1
```

> Any extra keys in `.env` (e.g. `ANTHROPIC_API_KEY`) are safely ignored.

### 5. Create a Pinecone dense index

In the [Pinecone console](https://app.pinecone.io) create a **serverless** index with:

| Setting | Value |
|---------|-------|
| Name | `ragpdfbot` (must match `PINECONE_INDEX_NAME`) |
| Dimensions | `1536` |
| Metric | `cosine` |
| Index type | **Dense** (not sparse) |
| Cloud / Region | `AWS us-east-1` (or match your `PINECONE_ENVIRONMENT`) |

> **Important:** The index must be **dense**. If you accidentally created a sparse index, delete it and recreate it as dense.

### 6. Start the FastAPI backend

```bash
python -m uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 7. Start the Streamlit frontend *(new terminal)*

```bash
python -m streamlit run streamlit_app.py
```

### 8. Open the app

Navigate to **http://localhost:8501** in your browser.

---

## Using the app

1. **Upload PDFs** — click the file uploader in the sidebar. Select one or more PDFs at once. Each one is processed and indexed automatically.
2. **See indexed documents** — the sidebar shows every uploaded file with its chunk count.
3. **Ask questions** — type in the chat box. Answers are generated from all uploaded PDFs combined.
4. **View sources** — expand the "Sources" section under any answer to see which file and page number the answer came from.
5. **Clear Chat** — removes conversation history but keeps PDFs indexed.
6. **Reset All** — clears everything including the uploaded file list (note: vectors remain in Pinecone).

---

## Project structure

```
rag-pdf-chat/
├── main.py                        # FastAPI app — CORS, lifespan, router
├── streamlit_app.py               # Streamlit chat UI — wide layout, multi-PDF
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template (no secrets)
├── .env                           # Your actual secrets (gitignored)
├── conftest.py                    # Pytest sys.path config
├── config/
│   └── config.py                  # pydantic-settings — loads .env, ignores extras
├── app/
│   ├── ingestion/
│   │   ├── pdf_loader.py          # PyMuPDF — extract text per page
│   │   └── chunker.py             # LangChain — 500-char chunks, 50-char overlap
│   ├── embeddings/
│   │   ├── embedder.py            # OpenAI — batch embed (50 chunks/call)
│   │   └── vector_store.py        # Pinecone — upsert (100/batch) and query
│   ├── rag/
│   │   ├── retriever.py           # Embed query → search Pinecone top-k
│   │   ├── generator.py           # Build prompt → call gpt-4o-mini
│   │   └── chain.py               # Orchestrate retrieve + generate
│   └── api/
│       ├── schemas.py             # Pydantic v2 request/response models
│       └── routes.py              # POST /api/upload, POST /api/ask
├── tests/
│   ├── test_ingestion.py          # Unit tests — PDF loading and chunking
│   └── test_rag.py                # Unit tests — RAG chain with mocks
└── uploads/                       # Saved PDF files (gitignored)
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Changelog

All changes made during setup and debugging are recorded here.

### Dependency fixes

| Issue | Fix |
|-------|-----|
| `pinecone-client==4.1.2` conflicted with `langchain-pinecone` | Removed `langchain-pinecone`, upgraded to `pinecone==6.0.2` |
| `Client.__init__() got unexpected keyword argument 'proxies'` | Upgraded `openai` to `1.55.3` and `httpx` to `0.28.1` |
| `anthropic 0.105.2` incompatible with `httpx==0.27.2` | Pinned `anthropic==0.49.0` (later removed entirely) |

### Configuration fixes

| Issue | Fix |
|-------|-----|
| `.env` copied from blank `.env.example` — all keys empty | Rewrote `.env` with actual key values via Python script |
| `Extra inputs are not permitted` — pydantic rejected unknown env vars | Changed `Settings` to use `model_config = {"extra": "ignore"}` |

### Pinecone index fixes

| Issue | Fix |
|-------|-----|
| `Upserting dense vectors is not supported for sparse indexes` | Deleted sparse index, recreated as dense (dim=1536, cosine) |

### Model / API fixes

| Issue | Fix |
|-------|-----|
| `model: claude-3-5-haiku-20241022` — 404 not found | Switched to OpenAI `gpt-4o-mini` (Anthropic removed entirely) |
| `Requested 503491 tokens, max 300000 per request` on large PDFs | Added batching to `embed_chunks()` — 50 chunks per API call |

### Feature additions

| Feature | Details |
|---------|---------|
| Multi-PDF upload | `accept_multiple_files=True` — each PDF indexed separately, no re-upload on rerun |
| Wide layout | `layout="wide"` + custom CSS for 1200px max content area |
| Cross-PDF querying | All PDFs share one Pinecone index — every question searches all documents |
| Indexed documents panel | Sidebar shows all uploaded files with chunk counts |
| Clear Chat / Reset All buttons | Separate controls for history vs full state reset |
| Answer streaming display | Answer rendered inline in the assistant bubble before `st.rerun()` |

---

## Skills demonstrated

- **Retrieval-Augmented Generation (RAG)** pipeline built from scratch
- **OpenAI API** — embeddings (`text-embedding-3-small`) and chat completions (`gpt-4o-mini`)
- **Pinecone vector database** — dense serverless index, batched upsert, cosine similarity search
- **PDF text extraction and semantic chunking** (PyMuPDF + LangChain)
- **REST API design** with FastAPI and Pydantic v2 schemas
- **Streamlit chat UI** — wide layout, session state, multi-file upload
- **Secure environment variable handling** with pydantic-settings
- **Dependency debugging** — resolving SDK conflicts across httpx, openai, pinecone
- **Unit testing** with pytest and unittest.mock

---

## License

MIT
