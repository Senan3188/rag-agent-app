# Project guidance

Applies to this repository and its subdirectories.

## Stack and architecture

- Use Python with `uv` for dependencies and commands; keep `uv.lock` updated when dependencies change.
- Keep the backend in FastAPI and the retrieval/answer workflow in LangGraph.
- Use local Ollama: `qwen3:8b` for answers and `nomic-embed-text` for embeddings by default. Keep models configurable.
- Keep the frontend plain HTML, CSS, and JavaScript in `app/static/`, served by FastAPI. No frontend build step is required.
- Store document chunks and embeddings in SQLite under the configured `DATA_DIR`. Original uploads are not retained.

## Code map

- `app/main.py`: app factory, lifespan, API routes, and static frontend serving.
- `app/config.py`: environment-based settings and validation.
- `app/documents.py`: text extraction and overlapping chunks.
- `app/store.py`: SQLite persistence and document-scoped cosine similarity search.
- `app/ollama.py`: asynchronous Ollama HTTP requests and response validation.
- `app/agent.py`: compiled LangGraph retrieve → answer workflow.
- `tests/test_api.py`: API, persistence, retrieval, extraction, and Ollama adapter tests.

## Development

```sh
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
uv run pytest -q
uv run ruff check .
node --check app/static/app.js
```

- The frontend is at `/`; interactive API documentation is at `/docs`.
- Live inference requires Ollama running and both configured models installed. Tests use a fake backend and need no running models.
- Run relevant checks after changes. For frontend changes, check JavaScript syntax and verify the page in a browser when available.
- Distinguish mocked verification from live Ollama verification in reports.

## Conventions

- Preserve the `create_app` factory and injectable Ollama backend for testing.
- Keep model I/O asynchronous; offload blocking extraction and SQLite work from async routes.
- Scope retrieval to the requested document. Preserve source text, citation numbers, and PDF page metadata.
- Treat uploaded text and model output as untrusted. Render them with `textContent`, not `innerHTML`.
- Validate inputs and return useful errors. Do not bypass upload limits or embedding model compatibility checks.
- Keep `.env`, uploaded content, databases, and credentials out of version control and logs.
- Preserve the local, single-user setup. Do not expose the server publicly or add cloud services without a request.
- Update `README.md` and `.env.example` when commands, API behavior, or configuration change.
