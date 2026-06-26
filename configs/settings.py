"""Application configuration via pydantic-settings."""
from __future__ import annotations
from enum import StrEnum
from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from models.domain import ChunkStrategy


class Environment(StrEnum):
    LOCAL = "local"; DEV = "dev"; STAGING = "staging"; PROD = "prod"


class LLMProvider(StrEnum):
    OLLAMA = "ollama"; OPENAI = "openai"; ANTHROPIC = "anthropic"; GEMINI = "gemini"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"; CRITICAL = "CRITICAL"


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    collection_name: str = "documents"
    # Qdrant Cloud: set api_key and https=true (managed clusters require TLS + auth).
    api_key: str | None = None
    https: bool = False

    @property
    def url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"


class EmbeddingSettings(BaseModel):
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    dimension: int = Field(default=768, ge=1)


class ChunkingSettings(BaseModel):
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    min_chunk_tokens: int = Field(default=32, ge=1)
    semantic_threshold_percentile: float = Field(default=95.0, ge=0.0, le=100.0)


class LLMSettings(BaseModel):
    provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    api_key: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    gemini_model: str = "gemini-2.0-flash"

class RetrievalSettings(BaseModel):
    top_k: int = Field(default=5, ge=1)
    top_k_dense: int = Field(default=20, ge=1)
    top_k_sparse: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    score_threshold: float | None = None
    multi_query_max: int = Field(default=3, ge=1)


class RerankerSettings(BaseModel):
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    top_k: int = Field(default=5, ge=1)


class GenerationSettings(BaseModel):
    gate_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_context_chunks: int = Field(default=5, ge=1)
    refusal_message: str = "I don't have enough information in the knowledge base to answer that confidently."


class CacheBackend(StrEnum):
    MEMORY = "memory"; REDIS = "redis"


class CacheSettings(BaseModel):
    enabled: bool = True
    backend: CacheBackend = CacheBackend.MEMORY
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = Field(default=3600, ge=1)


class LoggingSettings(BaseModel):
    log_level: LogLevel = LogLevel.INFO
    json_logs: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8",
        env_nested_delimiter="__", extra="ignore")
    app_name: str = "rag-engine"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    # Comma-separated list of allowed browser origins for CORS.
    # Default "*" allows any origin (convenient for demos); for production set
    # this to your frontend URL, e.g. RAG_CORS_ORIGINS=https://your-app.vercel.app
    cors_origins: str = "*"
    api: APISettings = Field(default_factory=APISettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
