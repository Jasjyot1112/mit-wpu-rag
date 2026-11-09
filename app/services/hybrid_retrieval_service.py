"""
Hybrid Retrieval Service combining dense (Pinecone) and sparse (BM25) search
with Reciprocal Rank Fusion (RRF).

Pipeline:
1. Generate query embedding
2. Search Pinecone (dense)
3. Search BM25 (sparse keyword)
4. Apply RRF to combine rankings
5. Deduplicate candidates
6. Return fused candidate list for reranking
"""
import logging
import time
from typing import Dict, List, Optional

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger

logger = get_logger("hybrid_retrieval")


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    score_keys: List[str],
    rrf_k: int = 60,
) -> List[Dict]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for item i: sum over lists of 1 / (rrf_k + rank_in_list(i))

    Args:
        ranked_lists: List of lists, each sorted by relevance (best first).
        score_keys: Key name for the score in each list dict (for preservation).
        rrf_k: RRF constant (default 60, standard from literature).

    Returns:
        Merged list sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = {}
    all_items: Dict[str, dict] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        score_key = score_keys[list_idx] if list_idx < len(score_keys) else "score"
        for rank, item in enumerate(ranked_list):
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if chunk_id not in all_items:
                all_items[chunk_id] = dict(item)
            # Preserve individual scores
            if score_key in item:
                all_items[chunk_id][score_key] = item[score_key]

    # Assign RRF scores and sort
    result = []
    for chunk_id, rrf_score in rrf_scores.items():
        item = all_items[chunk_id].copy()
        item["rrf_score"] = rrf_score
        result.append(item)

    result.sort(key=lambda x: x["rrf_score"], reverse=True)
    return result


class HybridRetrievalService:
    """
    Retrieval combining dense vector search (Pinecone) with BM25 keyword search,
    fused via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        embedding_service,
        vector_store,
        bm25_service,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        fusion_top_k: int = 30,
        final_top_k: int = 6,
        rrf_k: int = 60,
        similarity_threshold: float = 0.3,
    ):
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._bm25_service = bm25_service
        self._dense_top_k = dense_top_k
        self._bm25_top_k = bm25_top_k
        self._fusion_top_k = fusion_top_k
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        institution_id: str = "MIT-WPU",
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User query string.
            institution_id: Institution filter (mandatory).
            document_ids: Optional list to restrict search to specific documents.
            top_k: Override final_top_k if specified.

        Returns:
            dict with keys:
                candidates: List of chunk dicts after RRF
                dense_count: Number of dense results
                bm25_count: Number of BM25 results
                after_fusion: Number after RRF
                retrieval_time_ms: Total time
                query_embedding_time_ms: Embedding time
        """
        start_time = time.time()
        final_k = top_k if top_k is not None else self._final_top_k

        # --- Step 1: Generate query embedding ---
        embed_start = time.time()
        try:
            query_vector = self._embedding_service.embed_query(query)
        except Exception as exc:
            raise RetrievalError(f"Query embedding failed: {exc}")
        embed_time_ms = (time.time() - embed_start) * 1000

        # --- Step 2: Build Pinecone metadata filter ---
        pinecone_filter = {"institution_id": {"$eq": institution_id}}
        if document_ids:
            pinecone_filter["document_id"] = {"$in": document_ids}

        # --- Step 3: Dense search ---
        try:
            dense_results = self._vector_store.query_vectors(
                query_vector=query_vector,
                top_k=self._dense_top_k,
                filter_dict=pinecone_filter,
                include_metadata=True,
            )
        except Exception as exc:
            logger.warning("Dense search failed: %s. Falling back to BM25 only.", exc)
            dense_results = []

        # Convert Pinecone matches to standard format
        dense_chunks = []
        for match in dense_results:
            if match["score"] < self._similarity_threshold:
                continue
            meta = match.get("metadata", {})
            dense_chunks.append({
                "chunk_id": meta.get("chunk_id", match["id"]),
                "document_id": meta.get("document_id"),
                "document_name": meta.get("source_filename"),
                "source_type": meta.get("source_type"),
                "text": meta.get("text", ""),  # Note: text may not be in Pinecone metadata
                "heading": meta.get("heading"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "slide_start": meta.get("slide_start"),
                "slide_end": meta.get("slide_end"),
                "section": meta.get("section"),
                "dense_score": match["score"],
                "pinecone_id": match["id"],
            })

        # --- Step 4: BM25 search ---
        try:
            bm25_results = self._bm25_service.search(
                query=query,
                top_k=self._bm25_top_k,
                institution_id=institution_id,
                document_ids=document_ids,
            )
        except Exception as exc:
            logger.warning("BM25 search failed: %s. Proceeding with dense only.", exc)
            bm25_results = []

        # Rename BM25 score key for RRF
        for r in bm25_results:
            r["bm25_score"] = r.pop("bm25_score", 0.0)

        # --- Step 5: Reciprocal Rank Fusion ---
        fused = _reciprocal_rank_fusion(
            ranked_lists=[dense_chunks, bm25_results],
            score_keys=["dense_score", "bm25_score"],
            rrf_k=self._rrf_k,
        )

        # --- Step 6: Deduplicate by chunk_id ---
        seen_ids = set()
        deduplicated = []
        for item in fused:
            cid = item.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                deduplicated.append(item)

        # Limit to fusion_top_k before reranking
        candidates = deduplicated[:self._fusion_top_k]

        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Hybrid retrieval: dense=%d, bm25=%d, after_fusion=%d, final_candidates=%d (%.1fms)",
            len(dense_chunks), len(bm25_results), len(deduplicated), len(candidates), total_time_ms,
        )

        return {
            "candidates": candidates,
            "dense_count": len(dense_chunks),
            "bm25_count": len(bm25_results),
            "after_fusion": len(deduplicated),
            "retrieval_time_ms": total_time_ms,
            "query_embedding_time_ms": embed_time_ms,
        }
