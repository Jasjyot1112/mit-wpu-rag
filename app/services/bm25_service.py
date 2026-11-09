"""
BM25 keyword search service using rank-bm25.

Implements actual BM25 retrieval — not a simulation.
Persists index to disk and rebuilds from SQLite chunk metadata on startup.
"""
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger

logger = get_logger("bm25_service")

_DEFAULT_INDEX_PATH = "./storage/bm25_index.pkl"


class BM25Service:
    """
    BM25 keyword retrieval over indexed document chunks.

    The BM25 index stores tokenized chunk texts and maps back to chunk IDs.
    Index is persisted to disk and reconstructed from DB on startup if needed.
    """

    def __init__(self, index_path: str = _DEFAULT_INDEX_PATH):
        self._index_path = index_path
        self._bm25 = None
        self._chunk_ids: List[str] = []  # parallel array: index i → chunk_id
        self._chunk_texts: List[str] = []
        self._chunk_metadata: Dict[str, dict] = {}  # chunk_id → metadata
        self._is_built = False

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(
        self,
        chunks: List[dict],
        institution_id: str = "MIT-WPU",
    ) -> None:
        """
        Build (or rebuild) BM25 index from chunk records.

        Args:
            chunks: List of dicts with keys: chunk_id, text, and metadata fields.
            institution_id: Filter to this institution.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise RetrievalError("rank-bm25 not installed. Run: pip install rank-bm25")

        # Filter to institution
        relevant = [
            c for c in chunks
            if c.get("institution_id", institution_id) == institution_id
            and c.get("text", "").strip()
        ]

        if not relevant:
            logger.warning("No chunks available for BM25 index (institution: %s).", institution_id)
            self._bm25 = None
            self._chunk_ids = []
            self._is_built = False
            return

        self._chunk_ids = [c["chunk_id"] for c in relevant]
        self._chunk_texts = [c["text"] for c in relevant]
        self._chunk_metadata = {c["chunk_id"]: c for c in relevant}

        tokenized = [self._tokenize(text) for text in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._is_built = True

        logger.info("BM25 index built: %d chunks indexed.", len(relevant))
        self._save_index()

    def add_chunks(self, new_chunks: List[dict]) -> None:
        """
        Add new chunks to the existing index by rebuilding.
        (BM25Okapi doesn't support incremental updates; we rebuild.)
        """
        all_chunks = list(self._chunk_metadata.values()) + new_chunks
        institution_id = new_chunks[0].get("institution_id", "MIT-WPU") if new_chunks else "MIT-WPU"
        self.build_index(all_chunks, institution_id=institution_id)

    def remove_document(self, document_id: str) -> None:
        """
        Remove all chunks belonging to a document and rebuild the index.
        """
        remaining = [
            c for c in self._chunk_metadata.values()
            if c.get("document_id") != document_id
        ]
        if remaining:
            institution_id = remaining[0].get("institution_id", "MIT-WPU")
            self.build_index(remaining, institution_id=institution_id)
        else:
            self._bm25 = None
            self._chunk_ids = []
            self._chunk_texts = []
            self._chunk_metadata = {}
            self._is_built = False
            self._save_index()
        logger.info("Removed document '%s' from BM25 index.", document_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        institution_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            institution_id: Filter by institution (not currently used at BM25 level;
                            index is per-institution).
            document_ids: If set, restrict results to these document IDs.

        Returns:
            List of dicts: {chunk_id, text, score, metadata...}
        """
        if not self._is_built or self._bm25 is None:
            logger.warning("BM25 index not built; returning empty results.")
            return []

        if not query.strip():
            return []

        tokenized_query = self._tokenize(query)

        try:
            scores = self._bm25.get_scores(tokenized_query)
        except Exception as exc:
            raise RetrievalError(f"BM25 scoring failed: {exc}")

        # Pair (chunk_id, score)
        scored = [
            (self._chunk_ids[i], float(scores[i]))
            for i in range(len(self._chunk_ids))
        ]

        # Filter by document_ids if specified
        if document_ids:
            doc_set = set(document_ids)
            scored = [
                (cid, score) for cid, score in scored
                if self._chunk_metadata.get(cid, {}).get("document_id") in doc_set
            ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results = []
        token_set = set(tokenized_query)
        for chunk_id, score in top:
            metadata = self._chunk_metadata.get(chunk_id, {})
            doc_tokens = set(self._tokenize(metadata.get("text", "")))
            has_token_match = bool(token_set & doc_tokens)
            if score > 0.0 or has_token_match:
                effective_score = max(score, 0.01) if has_token_match and score <= 0.0 else score
                results.append({
                    "chunk_id": chunk_id,
                    "text": metadata.get("text", ""),
                    "bm25_score": effective_score,
                    "document_id": metadata.get("document_id"),
                    "document_name": metadata.get("source_filename"),
                    "source_type": metadata.get("source_type"),
                    "heading": metadata.get("heading"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "slide_start": metadata.get("slide_start"),
                    "slide_end": metadata.get("slide_end"),
                    "section": metadata.get("section"),
                })

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        """Public method to save index to disk."""
        self._save_index()

    def load_index(self) -> bool:
        """
        Load a previously saved BM25 index from disk.

        Returns:
            True if loaded successfully, False if no saved index exists.
        """
        if not os.path.exists(self._index_path):
            logger.info("No saved BM25 index found at '%s'.", self._index_path)
            return False

        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)

            self._bm25 = data["bm25"]
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data["chunk_texts"]
            self._chunk_metadata = data["chunk_metadata"]
            self._is_built = True

            logger.info(
                "BM25 index loaded from '%s': %d chunks.",
                self._index_path, len(self._chunk_ids),
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s. Will rebuild.", exc)
            return False

    def _save_index(self) -> None:
        """Save BM25 index state to disk using pickle."""
        try:
            data = {
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
                "chunk_texts": self._chunk_texts,
                "chunk_metadata": self._chunk_metadata,
            }
            with open(self._index_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("BM25 index saved to '%s'.", self._index_path)
        except Exception as exc:
            logger.error("Failed to save BM25 index: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenizer: lowercase, split on non-alphanumeric chars.
        Preserves technical terms (e.g., "BM25", "TCP/IP", "RSA").
        """
        import re
        # Lowercase
        text = text.lower()
        # Split on whitespace and common delimiters, but keep alphanumeric and dots/slashes
        tokens = re.findall(r"[a-z0-9]+(?:[./][a-z0-9]+)*", text)
        # Filter very short tokens (noise)
        return [t for t in tokens if len(t) > 1]
