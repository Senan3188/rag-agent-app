import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.ollama import Ollama, OllamaError


class FakeOllama:
    def __init__(self):
        self.messages = []

    async def embed(self, texts):
        return [[1.0, 0.0] if "Paris" in text else [0.0, 1.0] for text in texts]

    async def chat(self, messages):
        self.messages = messages
        return "The capital is Paris [1]."


@pytest.fixture
def client(tmp_path):
    backend = FakeOllama()
    app = create_app(Settings(data_dir=tmp_path), backend)
    with TestClient(app) as test_client:
        yield test_client, backend


def upload(client, text="The capital of France is Paris."):
    response = client.post("/documents", files={"file": ("facts.txt", text.encode())})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_upload_ask_and_delete(client):
    api, backend = client
    identifier = upload(api)
    response = api.post(f"/documents/{identifier}/ask", json={"question": "Where is Paris?"})
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "The capital is Paris [1]."
    assert response.json()["sources"][0]["citation"] == 1
    assert "Paris" in backend.messages[1]["content"]
    assert len(api.get("/documents").json()) == 1
    assert api.delete(f"/documents/{identifier}").status_code == 204
    assert api.get(f"/documents/{identifier}").status_code == 404


def test_isolation(client):
    api, backend = client
    first = upload(api, "Paris is in France.")
    upload(api, "Private unrelated content.")
    result = api.post(f"/documents/{first}/ask", json={"question": "Paris?"})
    assert result.status_code == 200
    assert "Private unrelated content" not in backend.messages[1]["content"]


@pytest.mark.parametrize(
    "name,content",
    [
        ("empty.txt", b""),
        ("bad.exe", b"abc"),
        ("bad.pdf", b"not a PDF"),
        ("bad.txt", b"\xff"),
        ("blank.txt", b"   "),
    ],
)
def test_invalid_files(client, name, content):
    api, _ = client
    assert api.post("/documents", files={"file": (name, content)}).status_code == 422
    assert api.get("/documents").json() == []


def test_unknown_and_invalid_questions(client):
    api, _ = client
    assert api.post("/documents/missing/ask", json={"question": "Hello?"}).status_code == 404
    identifier = upload(api)
    for payload in ({"question": " "}, {"question": "x", "top_k": 99}):
        assert api.post(f"/documents/{identifier}/ask", json=payload).status_code == 422


def test_size_limit(tmp_path):
    with TestClient(
        create_app(Settings(data_dir=tmp_path, max_upload_bytes=4), FakeOllama())
    ) as api:
        assert api.post("/documents", files={"file": ("x.txt", b"12345")}).status_code == 413


def test_persistence(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings, FakeOllama())) as api:
        identifier = upload(api)
    with TestClient(create_app(settings, FakeOllama())) as api:
        assert api.get(f"/documents/{identifier}").status_code == 200
        assert (
            api.post(f"/documents/{identifier}/ask", json={"question": "Paris?"}).status_code == 200
        )


def test_ollama_failure_does_not_save_document(client):
    api, backend = client

    async def fail(texts):
        raise OllamaError("Ollama is unavailable")

    backend.embed = fail
    assert api.post("/documents", files={"file": ("x.txt", b"Hello")}).status_code == 503
    assert api.get("/documents").json() == []


def test_embedding_model_change(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path), FakeOllama())) as api:
        identifier = upload(api)
    settings = Settings(data_dir=tmp_path, embedding_model="different")
    with TestClient(create_app(settings, FakeOllama())) as api:
        assert (
            api.post(f"/documents/{identifier}/ask", json={"question": "Paris?"}).status_code == 409
        )


def test_real_ollama_adapter_payloads():
    async def run():
        backend = Ollama(Settings())
        await backend.client.aclose()

        def respond(request):
            import json

            payload = json.loads(request.content)
            if request.url.path == "/api/embed":
                assert payload["model"] == "nomic-embed-text"
                return httpx.Response(200, json={"embeddings": [[1, 0]]})
            assert payload["model"] == "qwen3:8b"
            assert payload["think"] is False
            return httpx.Response(200, json={"message": {"content": "Paris [1]"}})

        backend.client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://ollama"
        )
        try:
            assert await backend.embed(["Paris"]) == [[1, 0]]
            assert await backend.chat([]) == "Paris [1]"
        finally:
            await backend.client.aclose()

    asyncio.run(run())


def test_retrieval_ranks_matching_chunk(tmp_path):
    from app.store import Store

    store = Store(tmp_path)
    chunks = [
        {"index": 0, "page": None, "text": "Other content"},
        {"index": 1, "page": 2, "text": "Paris"},
    ]
    doc = store.add("facts.pdf", "test", chunks, [[0, 1], [1, 0]])
    result = store.search(doc["id"], [1, 0], "test", 1)
    assert len(result) == 1
    assert result[0]["text"] == "Paris"
    assert result[0]["score"] == pytest.approx(1)


def test_pdf_extraction_with_page_metadata():
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from app.documents import extract_chunks

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Paris is in France.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    chunks = extract_chunks("facts.pdf", output.getvalue(), 1200, 200)
    assert chunks[0]["page"] == 1
    assert "Paris" in chunks[0]["text"]
