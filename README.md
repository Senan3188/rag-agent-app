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

Supports UTF-8 `.txt`, `.md`, `.csv` and text-based `.pdf` files. CSV files are parsed as structured tables with dynamic column names and comma, semicolon, tab, or pipe delimiters. A header row with unique, nonempty names is required. Scanned PDFs need OCR before upload.

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

PDF/text upload → text extraction → overlapping character chunks → Ollama embeddings → SQLite.

CSV upload → delimiter detection → validated headers and records → SQLite (no embeddings).

For each question, a compiled LangGraph `StateGraph` runs `retrieve → answer`. Retrieval embeds the question and ranks only that document's chunks by cosine similarity. The answer node sends retrieved excerpts to `qwen3:8b`, with instructions to cite evidence and acknowledge missing information. For CSVs, the graph instead runs `route → query_csv`. Simple questions referencing cell values search every column directly and pass complete matching candidate rows to the answer model; they do not require generated filters. Match metadata identifies the columns that matched, and ambiguous partial references require clarification. Counts, comparisons, sorting and numeric operations continue through the validated query planner. Qwen produces a schema-constrained query plan using column names and a few sample records. The planner receives exact and partial cell-value matches from every column, including short codes, numeric identifiers, email addresses, dates, and text, together with representative matching rows. No name or company columns are hardcoded. When an exact filter targets a column that lacks its value, the app corrects the column only if the mentioned value exists in exactly one column. Valid and ambiguous filters are preserved. The planner executes its proposal before answering and gets one bounded repair attempt for invalid queries or empty results when relevant values exist. If it cannot resolve the query, it requests clarification instead of asserting that matching data is absent. Python validates the plan and queries **all records**; Qwen then explains the computed result. It never executes generated Python or SQL. Supports case-insensitive exact/partial filters, numeric comparisons, counts, sum/average/min/max, sorting and selected columns. Filters are combined with AND. Unsupported or ambiguous requests ask for clarification. Select results default to 20 rows (maximum 100), with explicit match counts and truncation; aggregates use all matching rows. Numeric operations require plain numeric strings; currency and locale-specific numbers need normalization first. Blank aggregate cells are skipped. Source excerpts for CSVs show the executed plan and results with CSV record numbers (header is record 1). Duplicate matches are retained.

Questions are independent; no conversation history is retained. The chat model is separate from `nomic-embed-text`, the local embedding model.

SQLite in `DATA_DIR` stores extracted text, embeddings, and structured CSV records atomically. Existing databases gain a separate CSV table automatically. **Re-upload previously indexed CSV files** to use structured queries; old uploads return a clear migration message. No old documents are deleted. Original uploaded files are not retained. Retrieval uses NumPy exact search, suited to small local document collections. Changing the embedding model requires re-uploading PDF/text documents; structured CSVs do not depend on embeddings. All model requests go to the configured Ollama URL.

## Configuration

See `.env.example`. Defaults: Ollama at `http://localhost:11434`, 10 MiB upload limit, 1,200-character chunks with 200-character overlap, 2,000 chunks per document, and 180-second model timeout. `top_k` defaults to 4 and is limited to 8. Limit uploads further at a reverse proxy if deploying: the multipart parser may spool incoming request bodies before the endpoint applies its file limit.

The API binds to localhost in the run command. This is a local, single-user backend with no authentication or tenant isolation. Add authentication, request limits and deployment hardening before exposing it to other users. PDF parsing runs in a worker thread; use process isolation if accepting untrusted public uploads. Prompt instructions reduce but cannot eliminate hallucinations or document prompt injection.

## Development

```sh
uv run pytest
uv run ruff check .
```

Tests exercise uploads, retrieval and document isolation through the real LangGraph using a fake Ollama backend, plus the HTTP adapter's Ollama request format. They do not require installed models.

CSV query plans are validated for supported operations and column names, but the model can still misinterpret a question. Verify the plan and returned records under Supporting excerpts. Headerless files, duplicate headers and inconsistent record widths are rejected. This is not a general spreadsheet engine: joins, grouped aggregates and arbitrary formulas are not supported.
