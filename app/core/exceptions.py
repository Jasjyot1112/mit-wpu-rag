"""Custom exception hierarchy for the MIT-WPU RAG Engine."""

from __future__ import annotations


class RAGBaseException(Exception):
    """Base exception for all RAG Engine errors.

    Attributes:
        message: Human-readable description of the error.
        status_code: HTTP status code that should be returned to the client.
    """

    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"


# ---------------------------------------------------------------------------
# Document lifecycle errors
# ---------------------------------------------------------------------------

class DocumentNotFoundError(RAGBaseException):
    """Raised when a requested document cannot be found in the store."""

    status_code: int = 404


class DocumentAlreadyExistsError(RAGBaseException):
    """Raised when attempting to ingest a document that already exists."""

    status_code: int = 409


class InvalidDocumentError(RAGBaseException):
    """Raised when an uploaded document fails validation (corrupt, wrong format, etc.)."""

    status_code: int = 422


# ---------------------------------------------------------------------------
# Processing pipeline errors
# ---------------------------------------------------------------------------

class ExtractionError(RAGBaseException):
    """Raised when text or metadata extraction from a document fails."""

    status_code: int = 500


class EmbeddingError(RAGBaseException):
    """Raised when the embedding service (Gemini) fails to generate embeddings."""

    status_code: int = 502


class VectorStoreError(RAGBaseException):
    """Raised when communication with the vector store (Pinecone) fails."""

    status_code: int = 502


class RetrievalError(RAGBaseException):
    """Raised when the retrieval pipeline encounters an unrecoverable error."""

    status_code: int = 500


class GenerationError(RAGBaseException):
    """Raised when the LLM generation step (Gemini) fails."""

    status_code: int = 502


class InsufficientContextError(RAGBaseException):
    """Raised when no sufficiently relevant context can be found for a query.

    Note: status_code 200 because this is a valid, expected query outcome —
    not a server fault.
    """

    status_code: int = 200


# ---------------------------------------------------------------------------
# Auth / access errors
# ---------------------------------------------------------------------------

class AuthenticationError(RAGBaseException):
    """Raised when API key authentication fails."""

    status_code: int = 401


# ---------------------------------------------------------------------------
# File / upload errors
# ---------------------------------------------------------------------------

class FileSizeLimitError(RAGBaseException):
    """Raised when an uploaded file exceeds the configured size limit."""

    status_code: int = 413


class UnsupportedFileTypeError(RAGBaseException):
    """Raised when an uploaded file has a MIME type or extension not supported."""

    status_code: int = 415


# ---------------------------------------------------------------------------
# Job / worker errors
# ---------------------------------------------------------------------------

class JobNotFoundError(RAGBaseException):
    """Raised when a background processing job cannot be found."""

    status_code: int = 404


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimitError(RAGBaseException):
    """Raised when a client exceeds the configured request rate limit."""

    status_code: int = 429


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(RAGBaseException):
    """Raised when a required configuration value is invalid or missing at runtime."""

    status_code: int = 500
