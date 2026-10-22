import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import build_graph
from app.config import Settings
from app.csv_query import parse_csv
from app.documents import DocumentError, extract_chunks
from app.ollama import Ollama, OllamaError
from app.store import Store


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    top_k: int = Field(default=4, ge=1, le=8)


class Document(BaseModel):
    id: str
    filename: str
    created_at: str
    embedding_model: str
    chunk_count: int


class Source(BaseModel):
    citation: int
    index: int
    page: int | None
    text: str
    score: float
    kind: str = "excerpt"


class Answer(BaseModel):
    document_id: str
    answer: str
    sources: list[Source]


def create_app(settings: Settings | None = None, ollama=None):
    settings = settings or Settings()
    backend = ollama or Ollama(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(settings.data_dir)
        app.state.graph = build_graph(app.state.store, backend, settings)
        try:
            yield
        finally:
            if ollama is None:
                await backend.client.aclose()

    app = FastAPI(title="Local RAG Agent", version="0.1.0", lifespan=lifespan)

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def frontend():
        return FileResponse(static_dir / "index.html")

    @app.exception_handler(OllamaError)
    async def ollama_error(request: Request, exc: OllamaError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
        }

    @app.post("/documents", response_model=Document, status_code=201)
    async def upload(request: Request, file: Annotated[UploadFile, File()]):
        try:
            content = await file.read(settings.max_upload_bytes + 1)
        finally:
            await file.close()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "File exceeds MAX_UPLOAD_BYTES")
        if not content:
            raise HTTPException(422, "File is empty")
        filename = Path(file.filename or "upload.txt").name
        if Path(filename).suffix.lower() == ".csv":
            try:
                table = await asyncio.to_thread(parse_csv, content)
            except DocumentError as exc:
                raise HTTPException(422, str(exc)) from exc
            return await asyncio.to_thread(request.app.state.store.add_csv, filename, table)
        try:
            chunks = await asyncio.to_thread(
                extract_chunks, filename, content, settings.chunk_size, settings.chunk_overlap
            )
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        if len(chunks) > settings.max_chunks:
            raise HTTPException(413, "Document exceeds MAX_CHUNKS")
        vectors = await backend.embed([chunk["text"] for chunk in chunks])
        return await asyncio.to_thread(
            request.app.state.store.add, filename, settings.embedding_model, chunks, vectors
        )

    @app.get("/documents", response_model=list[Document])
    async def documents(request: Request):
        return await asyncio.to_thread(request.app.state.store.list)

    @app.get("/documents/{document_id}", response_model=Document)
    async def document(document_id: str, request: Request):
        try:
            return (await asyncio.to_thread(request.app.state.store.get, document_id))[0]
        except KeyError as exc:
            raise HTTPException(404, "Document not found") from exc

    @app.delete("/documents/{document_id}", status_code=204)
    async def delete(document_id: str, request: Request):
        if not await asyncio.to_thread(request.app.state.store.delete, document_id):
            raise HTTPException(404, "Document not found")

    @app.post("/documents/{document_id}/ask", response_model=Answer)
    async def ask(document_id: str, body: Question, request: Request):
        await document(document_id, request)
        try:
            result = await request.app.state.graph.ainvoke(
                {
                    "document_id": document_id,
                    "question": body.question,
                    "top_k": body.top_k,
                }
            )
        except KeyError as exc:
            raise HTTPException(404, "Document not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "document_id": document_id,
            "answer": result["answer"],
            "sources": [{"citation": i, **source} for i, source in enumerate(result["sources"], 1)],
        }

    return app


app = create_app()
