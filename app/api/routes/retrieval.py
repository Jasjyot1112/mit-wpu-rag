"""
Hybrid Retrieval Router.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.security import verify_api_key
from app.models.database import get_db
from app.repositories import chunk_repository
from app.schemas.retrieval_schemas import (
    RetrievalDiagnostics,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


def get_hybrid_retrieval(request: Request):
    return request.app.state.hybrid_retrieval_service


def get_reranker(request: Request):
    return request.app.state.reranking_service


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Execute hybrid dense + BM25 search with Reciprocal Rank Fusion",
)
def search_knowledge_base(
    request_data: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    retrieval_service=Depends(get_hybrid_retrieval),
    reranking_service=Depends(get_reranker),
):
    try:
        doc_ids = request_data.filters.document_ids if request_data.filters else None

        # 1. Hybrid retrieval (Pinecone + BM25 + RRF)
        res = retrieval_service.retrieve(
            query=request_data.query,
            institution_id=request_data.institution_id,
            document_ids=doc_ids,
            top_k=request_data.top_k,
        )

        candidates = res.get("candidates", [])

        # 2. Enrich text from SQLite for chunks that only have previews in Pinecone
        missing_ids = [c["chunk_id"] for c in candidates if not c.get("text")]
        if missing_ids:
            chunks_map = {
                c.id: c.text
                for c in db.query(chunk_repository.Chunk)
                .filter(chunk_repository.Chunk.id.in_(missing_ids))
                .all()
            }
            for c in candidates:
                if not c.get("text") and c["chunk_id"] in chunks_map:
                    c["text"] = chunks_map[c["chunk_id"]]

        # 3. Optional Reranking
        diagnostics_data = None
        rerank_time_ms = None
        if request_data.enable_reranking:
            candidates, rerank_diag = reranking_service.rerank(
                query=request_data.query,
                candidates=candidates,
                top_k=request_data.top_k,
            )
            rerank_time_ms = rerank_diag.get("reranking_time_ms")
        else:
            candidates = candidates[:request_data.top_k]

        if request_data.include_diagnostics:
            diagnostics_data = RetrievalDiagnostics(
                dense_candidates=res.get("dense_count", 0),
                bm25_candidates=res.get("bm25_count", 0),
                after_fusion=res.get("after_fusion", 0),
                after_reranking=len(candidates),
                reranking_enabled=request_data.enable_reranking,
                reranking_time_ms=rerank_time_ms,
                total_retrieval_time_ms=res.get("retrieval_time_ms", 0.0),
            )

        # 4. Map to response schema
        formatted_chunks: List[RetrievedChunk] = []
        for i, c in enumerate(candidates):
            formatted_chunks.append(
                RetrievedChunk(
                    chunk_id=c.get("chunk_id", ""),
                    document_id=c.get("document_id", ""),
                    document_name=c.get("document_name") or c.get("source_filename") or "",
                    source_type=c.get("source_type", ""),
                    text=c.get("text", ""),
                    heading=c.get("heading"),
                    page_start=c.get("page_start"),
                    page_end=c.get("page_end"),
                    slide_start=c.get("slide_start"),
                    slide_end=c.get("slide_end"),
                    dense_score=c.get("dense_score"),
                    bm25_score=c.get("bm25_score"),
                    rrf_score=c.get("rrf_score"),
                    rerank_score=c.get("rerank_score"),
                    final_rank=c.get("final_rank", i + 1),
                )
            )

        return SearchResponse(
            query=request_data.query,
            institution_id=request_data.institution_id,
            chunks=formatted_chunks,
            total_retrieved=len(formatted_chunks),
            diagnostics=diagnostics_data,
        )

    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
