"""
SQLAlchemy model for uploaded documents in the MIT-WPU RAG Engine.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, Index, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


def _utcnow() -> datetime:
    """Return current UTC time (timezone-naive for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Document(Base):
    """
    Represents a single document uploaded by an institution.

    Lifecycle
    ---------
    upload → processing_status: PENDING
           → processing_status: PROCESSING  (worker picks it up)
           → processing_status: COMPLETED / FAILED
           → indexing_status:   INDEXED / FAILED
    """

    __tablename__ = "documents"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    id: str = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False,
    )
    institution_id: str = Column(String, nullable=False, index=True)

    # ------------------------------------------------------------------
    # File metadata
    # ------------------------------------------------------------------
    filename: str = Column(String, nullable=False)          # original upload name
    safe_filename: str = Column(String, nullable=False)     # sanitised storage name
    file_extension: str = Column(String, nullable=True)     # .pdf / .pptx / .docx
    mime_type: str = Column(String, nullable=True)
    file_size: int = Column(Integer, nullable=True)         # bytes
    file_path: str = Column(String, nullable=True)          # relative storage path
    checksum: str = Column(String, nullable=True)           # SHA-256 of file content

    # ------------------------------------------------------------------
    # Upload timing
    # ------------------------------------------------------------------
    upload_timestamp: datetime = Column(
        DateTime, nullable=False, default=_utcnow
    )

    # ------------------------------------------------------------------
    # Processing status
    # processing_status values: PENDING | PROCESSING | COMPLETED | FAILED | REPROCESSING
    # ------------------------------------------------------------------
    processing_status: str = Column(
        String, nullable=False, default="PENDING"
    )
    processing_started_at: datetime | None = Column(DateTime, nullable=True)
    processing_completed_at: datetime | None = Column(DateTime, nullable=True)

    # ------------------------------------------------------------------
    # Extraction statistics
    # ------------------------------------------------------------------
    total_pages: int | None = Column(Integer, nullable=True)
    total_slides: int | None = Column(Integer, nullable=True)
    total_chunks: int = Column(Integer, nullable=False, default=0)
    extraction_method: str | None = Column(String, nullable=True)

    # ------------------------------------------------------------------
    # Embedding / indexing
    # embedding_model: name of the model used to embed chunks
    # indexing_status values: PENDING | INDEXED | FAILED | REINDEXING
    # ------------------------------------------------------------------
    embedding_model: str | None = Column(String, nullable=True)
    indexing_status: str = Column(
        String, nullable=False, default="PENDING"
    )

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------
    error_message: str | None = Column(String, nullable=True)

    # ------------------------------------------------------------------
    # Audit timestamps
    # ------------------------------------------------------------------
    created_at: datetime = Column(
        DateTime, nullable=False, default=_utcnow
    )
    updated_at: datetime = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    ingestion_jobs = relationship(
        "IngestionJob",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    extraction_errors = relationship(
        "ExtractionError",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Document id={self.id!r} filename={self.filename!r} "
            f"status={self.processing_status!r}>"
        )


# ---------------------------------------------------------------------------
# Composite index — common query pattern: "give me all PENDING docs for
# institution X"
# ---------------------------------------------------------------------------
Index(
    "ix_documents_institution_processing_status",
    Document.institution_id,
    Document.processing_status,
)
