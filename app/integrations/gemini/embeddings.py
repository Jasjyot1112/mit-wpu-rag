"""
Gemini Embedding Service using the official google-genai SDK.

Uses actual Gemini Embedding API — no fake/random vectors.
Implements:
- Batch embedding with configurable batch size
- Exponential backoff retry
- Vector dimension validation
- Task type configuration (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
"""
import logging
import time
from typing import List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger("embedding_service")


class EmbeddingService:
    """
    Generate embeddings using Gemini Embedding API.

    Supported task types:
    - RETRIEVAL_DOCUMENT  — for indexing document chunks
    - RETRIEVAL_QUERY     — for query embedding at retrieval time
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-004",
        dimension: int = 768,
        batch_size: int = 10,
        max_retries: int = 3,
        task_type_document: str = "RETRIEVAL_DOCUMENT",
        task_type_query: str = "RETRIEVAL_QUERY",
        timeout: int = 60,
    ):
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._task_type_document = task_type_document
        self._task_type_query = task_type_query
        self._timeout = timeout

        # Initialize the google-genai client
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=api_key)
            self._genai_types = types
            logger.info("Gemini embedding client initialized. Model: %s, Dimension: %d", model, dimension)
        except ImportError:
            raise EmbeddingError("google-genai SDK not installed. Run: pip install google-genai")
        except Exception as exc:
            raise EmbeddingError(f"Failed to initialize Gemini client: {exc}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks for indexing.

        Args:
            texts: List of non-empty text strings.

        Returns:
            List of embedding vectors (each of length self._dimension).

        Raises:
            EmbeddingError on API failure after retries.
        """
        return self._embed_batch(texts, task_type=self._task_type_document)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query for retrieval.

        Args:
            query: Query string.

        Returns:
            Embedding vector of length self._dimension.
        """
        if not query or not query.strip():
            raise EmbeddingError("Query text cannot be empty.")
        result = self._embed_batch([query], task_type=self._task_type_query)
        return result[0]

    def embed_texts(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Generic embedding with explicit task type."""
        return self._embed_batch(texts, task_type=task_type)

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Embed texts in batches. Returns all vectors in original order.
        """
        if not texts:
            return []

        # Validate inputs
        validated = []
        empty_indices = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.warning("Skipping empty text at index %d.", i)
                empty_indices.append(i)
            else:
                validated.append((i, text.strip()))

        if not validated:
            raise EmbeddingError("All provided texts are empty.")

        all_vectors: List[Tuple[int, List[float]]] = []

        # Process in batches
        for batch_start in range(0, len(validated), self._batch_size):
            batch = validated[batch_start: batch_start + self._batch_size]
            batch_indices = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]

            start_time = time.time()
            vectors = self._call_api_with_retry(batch_texts, task_type)
            elapsed = time.time() - start_time

            if len(vectors) != len(batch_texts):
                raise EmbeddingError(
                    f"API returned {len(vectors)} vectors for {len(batch_texts)} texts."
                )

            for orig_idx, vector in zip(batch_indices, vectors):
                self._validate_vector(vector, orig_idx)
                all_vectors.append((orig_idx, vector))

            logger.debug(
                "Embedded batch %d–%d (%d texts) in %.2fs.",
                batch_start, batch_start + len(batch), len(batch_texts), elapsed,
            )

            # Rate-limit courtesy pause between batches
            if batch_start + self._batch_size < len(validated):
                time.sleep(0.3)

        # Sort by original index and extract vectors
        all_vectors.sort(key=lambda x: x[0])
        result_vectors = [v for _, v in all_vectors]

        # Reinsert zero vectors for empty-text positions (they were skipped)
        # This keeps the output length matching the input length
        final = []
        result_iter = iter(result_vectors)
        for i in range(len(texts)):
            if i in empty_indices:
                final.append([0.0] * self._dimension)
            else:
                final.append(next(result_iter))

        return final

    def _call_api_with_retry(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Call the Gemini Embeddings API with exponential backoff retry."""
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                return self._call_api(texts, task_type)
            except Exception as exc:
                last_exception = exc
                if attempt < self._max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s
                    logger.warning(
                        "Embedding API call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempt + 1, self._max_retries, exc, wait_time,
                    )
                    time.sleep(wait_time)

        raise EmbeddingError(
            f"Embedding API failed after {self._max_retries} attempts: {last_exception}"
        )

    def _call_api(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Make a single API call to the Gemini Embeddings endpoint.
        Uses the official google-genai SDK.
        """
        try:
            # The google-genai SDK embed_content method
            response = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=self._genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimension,
                ),
            )
            # Response has .embeddings: list of ContentEmbedding objects
            vectors = [list(emb.values) for emb in response.embeddings]
            return vectors

        except Exception as exc:
            # Re-raise with context
            raise EmbeddingError(f"Gemini API call failed: {exc}") from exc

    def _validate_vector(self, vector: List[float], index: int) -> None:
        """Validate that the returned vector has the expected dimension."""
        if len(vector) != self._dimension:
            raise EmbeddingError(
                f"Vector at index {index} has dimension {len(vector)}, "
                f"expected {self._dimension}. "
                f"Check EMBEDDING_DIMENSION setting matches model capabilities."
            )
