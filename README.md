# Real Estate RAG Agent

LangGraph-powered RAG API for real estate PDFs. Upload lease documents, index them in Qdrant, and chat with an agent that routes between document Q&A and structured property extraction.

## Features

- PDF text extraction with OCR fallback for scanned pages
- Vector indexing with Google Gemini embeddings
- LangGraph supervisor routing (`document_qa` vs `property_extraction`)
- FastAPI endpoints for upload and chat

## Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDFs)
- [Poppler](https://poppler.freedesktop.org/) (required by `pdf2image` on Windows)
- Google Gemini API key

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy the environment template and add your API key:

```bash
copy .env.example .env
```

Set `GEMINI_API_KEY` in `.env`.

3. (Optional) Run Qdrant with Docker for persistent storage:

```bash
docker compose up -d
```

Then set in `.env`:

```
QDRANT_USE_MEMORY=false
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

By default, the app uses in-memory Qdrant so you can run without Docker.

## Run the API

```bash
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload and index a PDF |
| POST | `/chat` | Ask a question or extract property metadata |

## Test the pipeline

Generate a sample lease PDF and run the integration test (server must be running):

```bash
python generate_pdf.py
python test_pipeline.py
```

## Project structure

```
app/
  main.py              # FastAPI app
  config.py            # Environment settings
  agents/rag_graph.py  # LangGraph workflow
  services/
    pdf_parser.py      # PDF + OCR processing
    vector_store.py    # Qdrant indexing and retrieval
```
