"""
SQLAlchemy model for background ingestion / reprocessing jobs.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


def _utcnow() -> datetime:
    """Return current UTC time (timezone-naive for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class IngestionJob(Base):
    """
    Tracks a background worker job associated with a :class:`Document`.

    Job types
    ---------
    - **INGEST**     — initial extraction + embedding pipeline
    - **REPROCESS**  — re-run extraction / embedding for an existing document
    - **DELETE**     — remove chunks and vector records for a document

    Status transitions
    ------------------
    PENDING → RUNNING → COMPLETED
                      → FAILED  (retried up to max_attempts times)
    """

    __tablename__ = "ingestion_jobs"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    id: str = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Parent document linkage
    # ------------------------------------------------------------------
    document_id: str = Column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # Job configuration
    # job_type values: INGEST | REPROCESS | DELETE
    # status values:   PENDING | RUNNING | COMPLETED | FAILED
    # ------------------------------------------------------------------
    job_type: str = Column(String, nullable=False)
    status: str = Column(String, nullable=False, default="PENDING")

    # ------------------------------------------------------------------
    # Retry tracking
    # ------------------------------------------------------------------
    attempt_count: int = Column(Integer, nullable=False, default=0)
    max_attempts: int = Column(Integer, nullable=False, default=3)

    # ------------------------------------------------------------------
    # Diagnostic messages
    # ------------------------------------------------------------------
    error_message: str | None = Column(String, nullable=True)
    progress_message: str | None = Column(String, nullable=True)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    created_at: datetime = Column(
        DateTime, nullable=False, default=_utcnow
    )
    started_at: datetime | None = Column(DateTime, nullable=True)
    completed_at: datetime | None = Column(DateTime, nullable=True)
    updated_at: datetime = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    document = relationship(
        "Document",
        back_populates="ingestion_jobs",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IngestionJob id={self.id!r} type={self.job_type!r} "
            f"status={self.status!r} attempts={self.attempt_count}/{self.max_attempts}>"
        )
