import httpx
import numpy as np

from app.config import Settings


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"), timeout=settings.ollama_timeout
        )

    async def post(self, path, payload):
        try:
            response = await self.client.post(path, json=payload)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise OllamaError("Ollama reported an error; check its logs and installed models")
            return data
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(
                "Ollama request failed. Ensure Ollama is running and required models are pulled."
            ) from exc

    async def embed(self, texts):
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            data = await self.post(
                "/api/embed",
                {
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "truncate": False,
                },
            )
            result = data.get("embeddings", [])
            try:
                matrix = np.asarray(result, dtype=float)
                valid = (
                    matrix.ndim == 2
                    and matrix.shape[0] == len(batch)
                    and matrix.shape[1] > 0
                    and np.isfinite(matrix).all()
                    and (np.linalg.norm(matrix, axis=1) > 0).all()
                )
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise OllamaError("Ollama returned invalid embeddings")
            vectors.extend(result)
        return vectors

    async def chat(self, messages):
        data = await self.post(
            "/api/chat",
            {
                "model": self.settings.chat_model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_ctx": 16384},
            },
        )
        answer = data.get("message", {}).get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaError("Ollama returned an empty answer")
        return answer
