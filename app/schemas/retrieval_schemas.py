"""
Pydantic v2 schemas for vector/hybrid retrieval requests and responses.
MIT-WPU RAG Engine
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class MetadataFilter(BaseModel):
    """Optional metadata filters to narrow down retrieval candidates."""

    document_ids: Optional[List[str]] = Field(
        None, description="Restrict retrieval to these document UUIDs."
    )
    source_types: Optional[List[str]] = Field(
        None,
        description="Restrict retrieval to these source file types. Allowed values: PDF, PPTX, DOCX.",
    )


class SearchRequest(BaseModel):
    """Request body for the semantic / hybrid search endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language search query.")
    institution_id: str = Field(
        default="MIT-WPU", description="Institution namespace used to scope the search."
    )
    filters: Optional[MetadataFilter] = Field(
        None, description="Optional metadata filters applied before scoring."
    )
    top_k: int = Field(
        default=6, ge=1, le=20, description="Number of top-ranked chunks to return."
    )
    enable_reranking: bool = Field(
        default=True, description="Whether to apply a cross-encoder reranker after fusion."
    )
    include_diagnostics: bool = Field(
        default=False,
        description="When true, include detailed retrieval timing and candidate-count diagnostics.",
    )


# ---------------------------------------------------------------------------
# Response component schemas
# ---------------------------------------------------------------------------

class CitationSource(BaseModel):
    """Aggregated citation entry pointing to a source document and location."""

    document_id: str = Field(..., description="UUID of the source document.")
    document_name: str = Field(..., description="Human-readable document filename or title.")
    source_type: str = Field(..., description="File type of the source document, e.g. 'PDF'.")
    page_start: Optional[int] = Field(None, description="First page number referenced (1-indexed).")
    page_end: Optional[int] = Field(None, description="Last page number referenced (1-indexed).")
    slide_start: Optional[int] = Field(None, description="First slide number referenced (1-indexed).")
    slide_end: Optional[int] = Field(None, description="Last slide number referenced (1-indexed).")
    section: Optional[str] = Field(None, description="Section or chapter name, if available.")
    chunk_ids: List[str] = Field(..., description="UUIDs of all chunks from this source that were used.")
    relevance_score: Optional[float] = Field(
        None, description="Aggregate relevance score for this source (0–1)."
    )


class RetrievedChunk(BaseModel):
    """A single chunk returned from the retrieval pipeline with all scoring signals."""

    chunk_id: str = Field(..., description="UUID of the chunk.")
    document_id: str = Field(..., description="UUID of the parent document.")
    document_name: str = Field(..., description="Human-readable document filename or title.")
    source_type: str = Field(..., description="File type of the source document, e.g. 'PDF'.")
    text: str = Field(..., description="Raw text content of the chunk.")
    heading: Optional[str] = Field(None, description="Nearest section/slide heading, if detected.")
    page_start: Optional[int] = Field(None, description="First page number (1-indexed).")
    page_end: Optional[int] = Field(None, description="Last page number (1-indexed).")
    slide_start: Optional[int] = Field(None, description="First slide number (1-indexed).")
    slide_end: Optional[int] = Field(None, description="Last slide number (1-indexed).")
    dense_score: Optional[float] = Field(
        None, description="Cosine similarity score from the dense vector search."
    )
    bm25_score: Optional[float] = Field(
        None, description="BM25 lexical match score (normalised)."
    )
    rrf_score: Optional[float] = Field(
        None, description="Reciprocal Rank Fusion combined score."
    )
    rerank_score: Optional[float] = Field(
        None, description="Cross-encoder reranking score (present only when reranking is enabled)."
    )
    final_rank: Optional[int] = Field(
        None, description="1-based rank in the final returned result set."
    )


class RetrievalDiagnostics(BaseModel):
    """Diagnostic counters and timings for one retrieval call (returned when include_diagnostics=True)."""

    dense_candidates: int = Field(..., description="Number of chunks returned by the dense search stage.")
    bm25_candidates: int = Field(..., description="Number of chunks returned by the BM25 search stage.")
    after_fusion: int = Field(..., description="Number of unique chunks after Reciprocal Rank Fusion.")
    after_reranking: int = Field(..., description="Number of chunks after reranking and top-k truncation.")
    reranking_enabled: bool = Field(..., description="Whether the reranker was applied for this request.")
    reranking_time_ms: Optional[float] = Field(
        None, description="Wall-clock time spent on reranking, in milliseconds."
    )
    total_retrieval_time_ms: float = Field(
        ..., description="Total wall-clock time for the entire retrieval pipeline, in milliseconds."
    )


class SearchResponse(BaseModel):
    """Response body for the semantic / hybrid search endpoint."""

    query: str = Field(..., description="The original query string.")
    institution_id: str = Field(..., description="Institution namespace used for the search.")
    chunks: List[RetrievedChunk] = Field(..., description="Ranked list of retrieved chunks.")
    total_retrieved: int = Field(..., description="Total number of chunks returned (≤ top_k).")
    diagnostics: Optional[RetrievalDiagnostics] = Field(
        None, description="Diagnostic breakdown of the retrieval pipeline (present when include_diagnostics=True)."
    )
