# Project guidance

Applies to this repository and its subdirectories.

## Stack and architecture

- Use Python with `uv` for dependencies and commands; keep `uv.lock` updated when dependencies change.
- Keep the backend in FastAPI and the retrieval/answer workflow in LangGraph.
- Use local Ollama: `qwen3:8b` for answers and `nomic-embed-text` for embeddings by default. Keep models configurable.
- Keep the frontend plain HTML, CSS, and JavaScript in `app/static/`, served by FastAPI. No frontend build step is required.
- Store document chunks, embeddings, and structured CSV records in SQLite under `DATA_DIR`. Original uploads are not retained.

## Code map

- `app/main.py`: app factory, lifespan, API routes, and static frontend serving.
- `app/config.py`: environment-based settings and validation.
- `app/documents.py`: text extraction and overlapping chunks.
- `app/store.py`: SQLite persistence, structured CSV tables, and document-scoped cosine similarity search.
- `app/csv_query.py`: dynamic CSV parsing, all-column value search, complete candidate rows, validated plans, and deterministic queries.
- `app/ollama.py`: asynchronous Ollama HTTP requests and response validation.
- `app/agent.py`: LangGraph routing to PDF/text retrieval or structured CSV lookup/query and answer generation.
- `tests/test_api.py` and `tests/test_csv.py`: API, persistence, isolation, extraction, all-column lookup, planner repair, and adapter tests.

## Development

```sh
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
uv run pytest -q
uv run ruff check .
node --check app/static/app.js
```

- The frontend is at `/`; interactive API documentation is at `/docs`.
- Live answers need Ollama and the chat model; PDF/text indexing also needs the embedding model. Tests require no running models.
- Run relevant checks after changes. For frontend changes, check JavaScript syntax and verify the page in a browser when available.
- Distinguish mocked verification from live Ollama verification in reports.

## Conventions

- Preserve the `create_app` factory and injectable Ollama backend for testing.
- Keep model I/O asynchronous; offload blocking extraction and SQLite work from async routes.
- Query CSVs as structured data; never execute model-generated code or SQL. Never hardcode column names or dataset-specific entities.
- Simple CSV lookups search values across all columns and pass full candidate rows with matched-column metadata to the answer model. Preserve short codes, numeric IDs, leading zeros, dates, and emails.
- Prefer complete cell references over fragments inside them. Treat ambiguous partial matches as candidates; do not silently correct spelling or choose an identity.
- Counts, comparisons, sorting, and aggregates use validated plans over all matching rows. Keep planner repair bounded; do not drop valid conditions merely to produce results.
- Preserve duplicate matches, disclose truncation, and distinguish unresolved queries from confirmed zero matches. Old CSV text indexes require re-upload; preserve existing records.
- Scope retrieval to the requested document. Preserve source text, citation numbers, and PDF page metadata.
- Treat uploaded text and model output as untrusted. Render them with `textContent`, not `innerHTML`.
- Validate inputs and return useful errors. Do not bypass upload limits or embedding model compatibility checks.
- Keep `.env`, uploaded content, databases, and credentials out of version control and logs.
- Preserve the local, single-user setup. Do not expose the server publicly or add cloud services without a request.
- Update `README.md` and `.env.example` when commands, API behavior, or configuration change.
