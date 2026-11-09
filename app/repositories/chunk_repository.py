"""
Chunk Repository — SQLAlchemy sync CRUD operations for Chunks.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.chunk import Chunk


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_chunks(db: Session, chunks_data: List[Dict[str, Any]]) -> List[Chunk]:
    """Bulk create Chunk records and commit to database."""
    if not chunks_data:
        return []

    chunk_objects = []
    for item in chunks_data:
        data = dict(item)
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid4())
        data.setdefault("created_at", _now_utc())
        chunk_obj = Chunk(**data)
        chunk_objects.append(chunk_obj)

    db.add_all(chunk_objects)
    db.commit()
    for chunk_obj in chunk_objects:
        db.refresh(chunk_obj)
    return chunk_objects


def get_chunks_by_document(
    db: Session, document_id: str, skip: int = 0, limit: int = 100
) -> List[Chunk]:
    """Retrieve ordered chunks for a given document."""
    return (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_chunk(db: Session, chunk_id: str) -> Optional[Chunk]:
    """Retrieve a single chunk by its ID."""
    return db.query(Chunk).filter(Chunk.id == chunk_id).first()


def delete_chunks_by_document(db: Session, document_id: str) -> int:
    """Delete all chunks belonging to a document. Returns count of deleted chunks."""
    deleted_count = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .delete(synchronize_session="fetch")
    )
    db.commit()
    return deleted_count


def mark_chunks_indexed(
    db: Session, chunk_ids: List[str], vector_ids: List[str]
) -> None:
    """Mark chunks as indexed in vector store with their respective vector IDs."""
    if len(chunk_ids) != len(vector_ids):
        raise ValueError("chunk_ids and vector_ids must have identical length")

    for cid, vid in zip(chunk_ids, vector_ids):
        db.query(Chunk).filter(Chunk.id == cid).update(
            {"is_indexed": True, "vector_id": vid},
            synchronize_session="fetch",
        )
    db.commit()


def get_all_chunks_for_bm25(db: Session, institution_id: str) -> List[Dict[str, Any]]:
    """Retrieve all chunks for an institution formatted for BM25 indexing."""
    chunks = (
        db.query(Chunk)
        .filter(Chunk.institution_id == institution_id)
        .all()
    )
    result = []
    for c in chunks:
        result.append({
            "chunk_id": c.id,
            "document_id": c.document_id,
            "institution_id": c.institution_id,
            "source_filename": c.source_filename,
            "source_type": c.source_type,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "heading": c.heading,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "slide_start": c.slide_start,
            "slide_end": c.slide_end,
            "section": c.section,
            "extraction_method": c.extraction_method,
        })
    return result


def count_chunks(db: Session, document_id: str) -> int:
    """Return count of chunks for a document."""
    return db.query(Chunk).filter(Chunk.document_id == document_id).count()
