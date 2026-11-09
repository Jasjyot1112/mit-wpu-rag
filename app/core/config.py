"""Application configuration — reads from environment variables and .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised settings for the MIT-WPU RAG Engine.

    All fields are populated from environment variables (or a .env file).
    Fields with no default are **required** at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    APP_NAME: str = "MIT-WPU RAG Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # ------------------------------------------------------------ API Auth
    RAG_API_KEY: str = ""

    # ------------------------------------------------------------ Database
    DATABASE_URL: str = "sqlite:///./rag.db"

    # ------------------------------------------------------------- Storage
    UPLOAD_DIR: str = "./storage/uploads"
    MAX_FILE_SIZE_MB: int = 100
    MAX_PDF_PAGES: int = 500

    # -------------------------------------------------------------- Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_GENERATION_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-004"
    EMBEDDING_DIMENSION: int = 768
    GEMINI_EMBEDDING_TASK_TYPE: str = "RETRIEVAL_DOCUMENT"
    GEMINI_QUERY_TASK_TYPE: str = "RETRIEVAL_QUERY"
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_DELAY: float = 1.0
    GEMINI_BATCH_SIZE: int = 10
    GEMINI_TIMEOUT: int = 60

    # ------------------------------------------------------------ Pinecone
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "mit-wpu-rag"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_NAMESPACE: str = "mit-wpu"

    # ------------------------------------------------------------ Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # ---------------------------------------------------------- Retrieval
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    FUSION_TOP_K: int = 30
    FINAL_CONTEXT_TOP_K: int = 6
    SIMILARITY_THRESHOLD: float = 0.3
    RRF_K: int = 60

    # ----------------------------------------------------------------- OCR
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 250
    OCR_MIN_TEXT_LENGTH: int = 20

    # ---------------------------------------------------------- Reranking
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_MAX_CANDIDATES: int = 30

    # --------------------------------------------------------- Institution
    INSTITUTION_ID: str = "MIT-WPU"

    # ------------------------------------------------- Background workers
    MAX_WORKER_CONCURRENCY: int = 2
    JOB_RETRY_LIMIT: int = 3

    # -------------------------------------------------------- Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # ------------------------------------------------------------------
    def validate_startup(self) -> None:
        """Validate that all critical credentials are present.

        Raises:
            ValueError: If any required API key is missing or empty.
        """
        required: dict[str, str] = {
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "PINECONE_API_KEY": self.PINECONE_API_KEY,
            "RAG_API_KEY": self.RAG_API_KEY,
        }
        missing = [name for name, value in required.items() if not value or not value.strip()]
        if missing:
            raise ValueError(
                f"The following required environment variables are missing or empty: "
                f"{', '.join(missing)}. "
                f"Set them in your .env file or as environment variables before starting the server."
            )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    The cache is invalidated only when the process restarts, which is the
    correct behaviour for production.  In tests, call
    ``get_settings.cache_clear()`` before patching environment variables.
    """
    return Settings()
