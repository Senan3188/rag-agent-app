from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    data_dir: Path = Path("data")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    chunk_size: int = Field(default=1200, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    max_chunks: int = Field(default=2000, gt=0)
    ollama_timeout: float = Field(default=180, gt=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self
