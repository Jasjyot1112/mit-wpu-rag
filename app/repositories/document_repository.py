"""
Document Repository — SQLAlchemy sync CRUD operations for Documents,
IngestionJobs, and ExtractionErrors.
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.ingestion_job import IngestionJob
from app.models.extraction_error import ExtractionError
import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def create_document(db: Session, **kwargs) -> Document:
    """Create and persist a new Document record.

    Any keyword argument that matches a column on :class:`Document` can be
    supplied.  ``id`` is generated automatically when omitted.
    """
    if "id" not in kwargs or kwargs["id"] is None:
        kwargs["id"] = str(uuid.uuid4())

    kwargs.setdefault("created_at", _now_utc())
    kwargs.setdefault("updated_at", _now_utc())
    if "status" in kwargs and "processing_status" not in kwargs:
        kwargs["processing_status"] = kwargs.pop("status")
    kwargs.setdefault("processing_status", "PENDING")

    doc = Document(**kwargs)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, document_id: str) -> Optional[Document]:
    """Return the Document with *document_id*, or ``None`` if not found."""
    return db.query(Document).filter(Document.id == document_id).first()


def get_document_by_checksum(
    db: Session, checksum: str, institution_id: str
) -> Optional[Document]:
    """Look up a document by its SHA-256 checksum within an institution."""
    return (
        db.query(Document)
        .filter(
            Document.checksum == checksum,
            Document.institution_id == institution_id,
        )
        .first()
    )


def list_documents(
    db: Session,
    institution_id: str,
    skip: int = 0,
    limit: int = 50,
) -> List[Document]:
    """Return a paginated list of documents belonging to *institution_id*."""
    return (
        db.query(Document)
        .filter(Document.institution_id == institution_id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def count_documents(db: Session, institution_id: str) -> int:
    """Return the total number of documents for *institution_id*."""
    return (
        db.query(Document)
        .filter(Document.institution_id == institution_id)
        .count()
    )


def update_document_status(
    db: Session,
    document_id: str,
    status: str,
    **kwargs,
) -> Optional[Document]:
    """Update the ``status`` field (and any extra *kwargs*) on a Document.

    Returns the updated Document or ``None`` if not found.
    """
    doc = get_document(db, document_id)
    if doc is None:
        return None

    doc.processing_status = status
    doc.updated_at = _now_utc()

    for key, value in kwargs.items():
        if hasattr(doc, key):
            setattr(doc, key, value)

    db.commit()
    db.refresh(doc)
    return doc


def update_document_processing_complete(
    db: Session,
    document_id: str,
    total_chunks: int,
    extraction_method: str,
    embedding_model: str,
) -> Optional[Document]:
    """Mark a document as COMPLETED and record processing metadata."""
    doc = get_document(db, document_id)
    if doc is None:
        return None

    doc.processing_status = "COMPLETED"
    doc.indexing_status = "INDEXED"
    doc.total_chunks = total_chunks
    doc.extraction_method = extraction_method
    doc.embedding_model = embedding_model
    doc.processing_completed_at = _now_utc()
    doc.updated_at = _now_utc()

    db.commit()
    db.refresh(doc)
    return doc


def delete_document(db: Session, document_id: str) -> bool:
    """Delete a Document and return ``True`` on success, ``False`` if not found."""
    doc = get_document(db, document_id)
    if doc is None:
        return False

    db.delete(doc)
    db.commit()
    return True


# ---------------------------------------------------------------------------
# IngestionJob operations
# ---------------------------------------------------------------------------

def create_ingestion_job(
    db: Session, document_id: str, job_type: str = "INGEST"
) -> IngestionJob:
    """Create a new IngestionJob for *document_id* with status PENDING."""
    job = IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        job_type=job_type,
        status="PENDING",
        attempt_count=0,
        max_attempts=3,
        created_at=_now_utc(),
        updated_at=_now_utc(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job_status(
    db: Session,
    job_id: str,
    status: str,
    error_message: str = None,
    progress_message: str = None,
) -> Optional[IngestionJob]:
    """Update the status of an IngestionJob.

    Automatically records *started_at* when transitioning to RUNNING and
    *completed_at* / *failed_at* on terminal states.
    """
    job = get_job(db, job_id)
    if job is None:
        return None

    job.status = status
    job.updated_at = _now_utc()

    if status == "RUNNING":
        job.started_at = _now_utc()
        job.attempt_count = (job.attempt_count or 0) + 1
    elif status in ("COMPLETED", "FAILED", "CANCELLED"):
        job.completed_at = _now_utc()

    if error_message is not None:
        job.error_message = error_message

    if progress_message is not None:
        job.progress_message = progress_message

    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[IngestionJob]:
    """Return the IngestionJob with *job_id*, or ``None``."""
    return db.query(IngestionJob).filter(IngestionJob.id == job_id).first()


def get_pending_jobs(db: Session, limit: int = 10) -> List[IngestionJob]:
    """Return up to *limit* jobs whose status is PENDING, ordered by creation time."""
    return (
        db.query(IngestionJob)
        .filter(IngestionJob.status == "PENDING")
        .order_by(IngestionJob.created_at.asc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# ExtractionError operations
# ---------------------------------------------------------------------------

def create_extraction_error(
    db: Session,
    document_id: str,
    page_number: Optional[int],
    slide_number: Optional[int],
    error_type: str,
    error_message: str,
) -> ExtractionError:
    """Record a non-fatal extraction error for a document page/slide."""
    err = ExtractionError(
        id=str(uuid.uuid4()),
        document_id=document_id,
        page_number=page_number,
        slide_number=slide_number,
        error_type=error_type,
        error_message=error_message,
        created_at=_now_utc(),
    )
    db.add(err)
    db.commit()
    db.refresh(err)
    return err


def get_extraction_errors(
    db: Session, document_id: str
) -> List[ExtractionError]:
    """Return all ExtractionErrors recorded for *document_id*."""
    return (
        db.query(ExtractionError)
        .filter(ExtractionError.document_id == document_id)
        .order_by(ExtractionError.created_at.asc())
        .all()
    )
