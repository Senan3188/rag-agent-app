# Local RAG Agent

A backend for uploading a document and asking questions about it. Built with **uv**, **FastAPI**, **LangGraph**, and local **Ollama**. No cloud API keys or external vector database are required.

## Run

Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), then:

```sh
ollama serve  # skip if the Ollama desktop app is already running
```

In another terminal:

```sh
ollama pull qwen3:8b
ollama pull nomic-embed-text
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the HTML/JavaScript frontend: upload a file, choose a saved document, and ask questions with expandable source excerpts. No frontend build step is needed. The interactive API is at http://127.0.0.1:8000/docs. Python 3.11+ is required.

## Upload and ask

Supports UTF-8 `.txt`, `.md`, `.csv` and text-based `.pdf` files. CSV is treated as text, not as a queryable database. Scanned PDFs need OCR before upload.

```sh
curl -X POST http://127.0.0.1:8000/documents \
  -F 'file=@/path/to/document.pdf'
```

Copy the returned `id` and use it below:

```sh
curl -X POST http://127.0.0.1:8000/documents/DOCUMENT_ID/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the main points?", "top_k":4}'
```

Responses include `answer` and ranked `sources`. Answer citations such as `[1]` correspond to each source's `citation` field. Each source includes chunk text, a zero-based chunk index, a PDF page number (or null), and cosine similarity. Sources are retrieved evidence; model-generated citations are not mechanically verified.

Other endpoints:

- `GET /health`: API liveness and configured models; does not probe Ollama readiness.
- `GET /documents`: list uploaded documents.
- `GET /documents/{id}`: document metadata.
- `DELETE /documents/{id}`: remove a document and its index.

## How it works

Upload → text extraction → overlapping character chunks → Ollama embeddings → SQLite.

For each question, a compiled LangGraph `StateGraph` runs `retrieve → answer`. Retrieval embeds the question and ranks only that document's chunks by cosine similarity. The answer node sends retrieved excerpts to `qwen3:8b`, with instructions to cite evidence and acknowledge missing information. Questions are independent; no conversation history is retained. The chat model is separate from `nomic-embed-text`, the local embedding model.

SQLite in `DATA_DIR` stores extracted text and embeddings atomically. Original uploaded files are not retained. Retrieval uses NumPy exact search, suited to small local document collections. Changing the embedding model requires re-uploading documents. All model requests go to the configured Ollama URL.

## Configuration

See `.env.example`. Defaults: Ollama at `http://localhost:11434`, 10 MiB upload limit, 1,200-character chunks with 200-character overlap, 2,000 chunks per document, and 180-second model timeout. `top_k` defaults to 4 and is limited to 8. Limit uploads further at a reverse proxy if deploying: the multipart parser may spool incoming request bodies before the endpoint applies its file limit.

The API binds to localhost in the run command. This is a local, single-user backend with no authentication or tenant isolation. Add authentication, request limits and deployment hardening before exposing it to other users. PDF parsing runs in a worker thread; use process isolation if accepting untrusted public uploads. Prompt instructions reduce but cannot eliminate hallucinations or document prompt injection.

## Development

```sh
uv run pytest
uv run ruff check .
```

Tests exercise uploads, retrieval and document isolation through the real LangGraph using a fake Ollama backend, plus the HTTP adapter's Ollama request format. They do not require installed models.
