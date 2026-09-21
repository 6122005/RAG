"""Application configuration module using pydantic-settings."""

import os
from functools import lru_cache
from pathlib import Path

# Ensure HuggingFace cache resides on D: drive and uses local cache instantly
os.environ.setdefault("HF_HOME", r"D:\.cache\huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application configuration settings."""

    # API Security
    API_KEY: str = "rag-secret-key-prod-2026"

    # LLM Settings
    LLM_PROVIDER: str = "openrouter"  # "openrouter", "gemini", "openai", or "mock"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_FALLBACK_MODEL: str = "google/gemini-2.0-flash-exp:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.2

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "sentence-transformers"  # "sentence-transformers", "gemini", "openai"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Reranking & Retrieval Settings
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_THRESHOLD: float = 0.001
    TOP_K_RETRIEVAL: int = 10
    TOP_N_RERANK: int = 4

    # Storage Paths
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    DOCUMENT_STORE_DIR: str = "./data/documents"

    # Server & Rate Limiting
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RATE_LIMIT_PER_MINUTE: int = 60

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
