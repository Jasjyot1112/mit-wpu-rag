"""
Pydantic v2 schemas for document upload, processing, and chunk responses.
MIT-WPU RAG Engine
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REPROCESSING = "REPROCESSING"


class IndexingStatus(str, Enum):
    PENDING = "PENDING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    REINDEXING = "REINDEXING"


class SourceType(str, Enum):
    PDF = "PDF"
    PPTX = "PPTX"
    DOCX = "DOCX"


# ---------------------------------------------------------------------------
# Document response schemas
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    """Full representation of a document record returned from the API."""

    id: str = Field(..., description="UUID of the document record.")
    institution_id: str = Field(..., description="Owning institution identifier.")
    filename: str = Field(..., description="Original filename as uploaded.")
    file_extension: str = Field(..., description="Lowercase file extension without dot, e.g. 'pdf'.")
    mime_type: str = Field(..., description="MIME type detected at upload time.")
    file_size: int = Field(..., description="File size in bytes.")
    upload_timestamp: datetime = Field(..., description="UTC timestamp when the file was received.")
    processing_status: ProcessingStatus = Field(..., description="Current text-extraction pipeline status.")
    total_pages: Optional[int] = Field(None, description="Total number of pages (PDF / DOCX).")
    total_slides: Optional[int] = Field(None, description="Total number of slides (PPTX).")
    total_chunks: Optional[int] = Field(None, description="Number of text chunks produced after processing.")
    extraction_method: Optional[str] = Field(
        None,
        description="Method used for text extraction, e.g. 'native', 'ocr', 'hybrid'.",
    )
    embedding_model: Optional[str] = Field(
        None,
        description="Identifier of the embedding model used for this document.",
    )
    indexing_status: IndexingStatus = Field(..., description="Current vector-index status.")
    error_message: Optional[str] = Field(None, description="Human-readable error if processing/indexing failed.")
    created_at: datetime = Field(..., description="UTC timestamp when the database record was created.")

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: List[DocumentResponse]
    total: int = Field(..., description="Total number of documents matching the query (before pagination).")


class DocumentStatusResponse(BaseModel):
    """Lightweight status-only view of a document, used for polling."""

    id: str = Field(..., description="UUID of the document record.")
    processing_status: ProcessingStatus
    indexing_status: IndexingStatus
    total_chunks: Optional[int] = None
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = Field(
        None, description="UTC timestamp when processing was initiated."
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="UTC timestamp when processing finished (success or failure)."
    )

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Immediate response returned after a document upload is accepted."""

    id: str = Field(..., description="UUID assigned to the new document record.")
    filename: str = Field(..., description="Stored filename.")
    processing_status: ProcessingStatus
    job_id: str = Field(..., description="Background-job identifier that can be used to track progress.")
    message: str = Field(..., description="Human-readable confirmation message.")


# ---------------------------------------------------------------------------
# Chunk response schemas
# ---------------------------------------------------------------------------

class ChunkResponse(BaseModel):
    """Full representation of a single text chunk."""

    id: str = Field(..., description="UUID of the chunk record.")
    document_id: str = Field(..., description="UUID of the parent document.")
    source_filename: str = Field(..., description="Original filename of the source document.")
    source_type: SourceType = Field(..., description="File type of the source document.")
    chunk_index: int = Field(..., description="Zero-based position of this chunk within the document.")
    text: str = Field(..., description="Raw text content of the chunk.")
    heading: Optional[str] = Field(None, description="Nearest section/slide heading, if detected.")
    page_start: Optional[int] = Field(None, description="First page number covered by this chunk (1-indexed).")
    page_end: Optional[int] = Field(None, description="Last page number covered by this chunk (1-indexed).")
    slide_start: Optional[int] = Field(None, description="First slide number covered by this chunk (1-indexed).")
    slide_end: Optional[int] = Field(None, description="Last slide number covered by this chunk (1-indexed).")
    section: Optional[str] = Field(None, description="Document section or chapter name, if available.")
    extraction_method: Optional[str] = Field(
        None, description="Method used to extract text for this chunk, e.g. 'native', 'ocr'."
    )
    token_count: Optional[int] = Field(None, description="Approximate token count of the chunk text.")
    is_indexed: bool = Field(..., description="Whether this chunk has been successfully indexed in the vector store.")
    created_at: datetime = Field(..., description="UTC timestamp when the chunk record was created.")

    model_config = {"from_attributes": True}


class ChunksListResponse(BaseModel):
    """Paginated list of chunks belonging to a specific document."""

    chunks: List[ChunkResponse]
    total: int = Field(..., description="Total number of chunks for the document (before pagination).")
    document_id: str = Field(..., description="UUID of the parent document.")
