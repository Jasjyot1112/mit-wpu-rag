"""
Pinecone Vector Store integration.

Uses actual Pinecone serverless index — no simulated results.
Implements:
- Index creation / validation
- Upsert with metadata
- Vector search with metadata filtering
- Document vector deletion
- Idempotent operations
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger("pinecone.vector_store")

_PINECONE_METADATA_VALUE_MAX_LEN = 512  # Pinecone metadata string limit


class PineconeVectorStore:
    """
    Real Pinecone vector store for RAG chunk indexing and retrieval.

    All upsert and query operations hit the actual Pinecone API.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        cloud: str,
        region: str,
        dimension: int = 768,
        namespace: str = "mit-wpu",
        max_retries: int = 3,
    ):
        self._index_name = index_name
        self._cloud = cloud
        self._region = region
        self._dimension = dimension
        self._namespace = namespace
        self._max_retries = max_retries
        self._index = None

        try:
            from pinecone import Pinecone, ServerlessSpec
            self._pc = Pinecone(api_key=api_key)
            self._ServerlessSpec = ServerlessSpec
            logger.info("Pinecone client initialized. Index: %s", index_name)
        except ImportError:
            raise VectorStoreError("pinecone package not installed. Run: pip install pinecone")
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Pinecone client: {exc}")

    def initialize(self) -> None:
        """
        Create the index if it doesn't exist and validate dimension.
        Must be called at application startup.
        """
        try:
            existing_indexes = [idx.name for idx in self._pc.list_indexes()]

            if self._index_name not in existing_indexes:
                logger.info("Creating Pinecone index '%s' (dim=%d)...", self._index_name, self._dimension)
                self._pc.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric="cosine",
                    spec=self._ServerlessSpec(
                        cloud=self._cloud,
                        region=self._region,
                    ),
                )
                # Wait for index to be ready
                self._wait_for_index_ready()
                logger.info("Pinecone index '%s' created successfully.", self._index_name)
            else:
                logger.info("Pinecone index '%s' already exists.", self._index_name)
                # Validate dimension
                index_info = self._pc.describe_index(self._index_name)
                actual_dim = index_info.dimension
                if actual_dim != self._dimension:
                    raise VectorStoreError(
                        f"Pinecone index dimension mismatch: index has {actual_dim}, "
                        f"configured EMBEDDING_DIMENSION is {self._dimension}. "
                        f"Either update EMBEDDING_DIMENSION or delete and recreate the index."
                    )

            self._index = self._pc.Index(self._index_name)
            stats = self._index.describe_index_stats()
            logger.info(
                "Pinecone index '%s' ready. Total vectors: %d",
                self._index_name, stats.total_vector_count,
            )

        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Pinecone initialization failed: {exc}")

    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
    ) -> int:
        """
        Upsert vectors into Pinecone in batches.

        Args:
            vectors: List of (vector_id, embedding, metadata) tuples.

        Returns:
            Number of vectors upserted.
        """
        self._ensure_initialized()
        if not vectors:
            return 0

        total_upserted = 0
        # Pinecone recommends batches of up to 100 vectors
        batch_size = 100

        for batch_start in range(0, len(vectors), batch_size):
            batch = vectors[batch_start: batch_start + batch_size]
            pinecone_records = []

            for vector_id, embedding, metadata in batch:
                # Sanitize metadata (truncate long strings)
                safe_metadata = self._sanitize_metadata(metadata)
                pinecone_records.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": safe_metadata,
                })

            self._upsert_with_retry(pinecone_records)
            total_upserted += len(batch)

        logger.info("Upserted %d vectors to Pinecone index '%s'.", total_upserted, self._index_name)
        return total_upserted

    def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for nearest neighbors.

        Args:
            query_vector: Dense embedding of the query.
            top_k: Number of results to return.
            filter_dict: Pinecone metadata filter (e.g., {"institution_id": {"$eq": "MIT-WPU"}}).
            include_metadata: Whether to include metadata in results.

        Returns:
            List of match dicts with keys: id, score, metadata.
        """
        self._ensure_initialized()

        try:
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "namespace": self._namespace,
                "include_metadata": include_metadata,
                "include_values": False,
            }
            if filter_dict:
                query_params["filter"] = filter_dict

            response = self._index.query(**query_params)
            matches = []
            for match in response.matches:
                matches.append({
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": match.metadata or {},
                })
            return matches

        except Exception as exc:
            raise VectorStoreError(f"Pinecone query failed: {exc}")

    def delete_vectors_by_document(self, document_id: str) -> None:
        """
        Delete all vectors for a given document_id using metadata filter.
        """
        self._ensure_initialized()
        try:
            # Pinecone serverless supports delete by filter
            self._index.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=self._namespace,
            )
            logger.info("Deleted vectors for document_id='%s' from Pinecone.", document_id)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete vectors for document '{document_id}': {exc}"
            )

    def delete_vectors_by_ids(self, vector_ids: List[str]) -> None:
        """Delete specific vectors by their IDs."""
        self._ensure_initialized()
        if not vector_ids:
            return
        try:
            # Batch delete in groups of 1000
            for i in range(0, len(vector_ids), 1000):
                batch = vector_ids[i: i + 1000]
                self._index.delete(ids=batch, namespace=self._namespace)
            logger.info("Deleted %d vectors by ID.", len(vector_ids))
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete vectors: {exc}")

    def get_index_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        self._ensure_initialized()
        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": {
                    ns: {"vector_count": data.vector_count}
                    for ns, data in (stats.namespaces or {}).items()
                },
            }
        except Exception as exc:
            raise VectorStoreError(f"Failed to get index stats: {exc}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._index is None:
            raise VectorStoreError(
                "Pinecone index not initialized. Call initialize() first."
            )

    def _upsert_with_retry(self, records: List[dict]) -> None:
        """Upsert records with retry on transient failures."""
        for attempt in range(self._max_retries):
            try:
                self._index.upsert(vectors=records, namespace=self._namespace)
                return
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Pinecone upsert failed (attempt %d/%d): %s. Retrying in %ds.",
                        attempt + 1, self._max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise VectorStoreError(f"Pinecone upsert failed after {self._max_retries} attempts: {exc}")

    def _wait_for_index_ready(self, timeout: int = 120) -> None:
        """Poll until the index is ready to serve requests."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                info = self._pc.describe_index(self._index_name)
                if info.status.ready:
                    return
            except Exception:
                pass
            time.sleep(5)
        raise VectorStoreError(
            f"Pinecone index '{self._index_name}' did not become ready within {timeout}s."
        )

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure metadata values are safe for Pinecone.
        - Truncate long strings.
        - Remove None values.
        - Convert non-primitive types.
        """
        safe = {}
        for key, value in metadata.items():
            if value is None:
                continue  # Pinecone doesn't store None
            if isinstance(value, str):
                safe[key] = value[:_PINECONE_METADATA_VALUE_MAX_LEN]
            elif isinstance(value, (int, float, bool)):
                safe[key] = value
            elif isinstance(value, list):
                # Only keep list of strings/numbers
                safe[key] = [
                    str(v)[:100] if isinstance(v, str) else v
                    for v in value
                    if isinstance(v, (str, int, float, bool))
                ][:20]  # Cap list length
            else:
                safe[key] = str(value)[:_PINECONE_METADATA_VALUE_MAX_LEN]
        return safe
