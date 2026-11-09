"""
SQLAlchemy model for page/slide-level extraction errors recorded during
the ingestion pipeline.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


def _utcnow() -> datetime:
    """Return current UTC time (timezone-naive for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ExtractionError(Base):
    """
    Records a single extraction failure encountered while processing a
    :class:`Document`.

    Multiple errors may be recorded for a single document (e.g. one per
    page/slide that fails OCR).  The ingestion pipeline continues past
    individual page failures and records them here for later analysis or
    manual review.

    error_type examples
    -------------------
    - ``OCR_FAILURE``        — Tesseract / OCR backend failed
    - ``EXTRACTION_FAILURE`` — PDF/PPTX parser raised an exception
    - ``ENCODING_ERROR``     — text encoding could not be determined
    - ``RENDERING_ERROR``    — page could not be rendered to image
    - ``UNKNOWN``            — catch-all for unexpected errors
    """

    __tablename__ = "extraction_errors"

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
    # Location within document
    # ------------------------------------------------------------------
    page_number: int | None = Column(Integer, nullable=True)
    slide_number: int | None = Column(Integer, nullable=True)

    # ------------------------------------------------------------------
    # Error details
    # ------------------------------------------------------------------
    error_type: str = Column(String, nullable=False)
    error_message: str = Column(String, nullable=False)
    recovery_attempted: bool = Column(Boolean, nullable=False, default=False)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    created_at: datetime = Column(
        DateTime, nullable=False, default=_utcnow
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    document = relationship(
        "Document",
        back_populates="extraction_errors",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ExtractionError id={self.id!r} document_id={self.document_id!r} "
            f"type={self.error_type!r} page={self.page_number} slide={self.slide_number}>"
        )
