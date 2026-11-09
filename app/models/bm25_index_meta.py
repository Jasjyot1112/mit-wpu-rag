"""
SQLAlchemy model for tracking persisted BM25 (sparse retrieval) index metadata.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from .database import Base


def _utcnow() -> datetime:
    """Return current UTC time (timezone-naive for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class BM25IndexMeta(Base):
    """
    Stores metadata about a persisted BM25 index for a given institution.

    The BM25 index itself is serialised to disk (``index_path``) by the
    background worker after documents are ingested.  This table lets the
    retrieval service quickly determine whether a valid index exists before
    attempting to load it, and allows the worker to invalidate stale
    indexes (``is_valid = False``) when new documents are added.

    One row per institution per rebuild — the latest valid row for a given
    ``institution_id`` is authoritative.
    """

    __tablename__ = "bm25_index_meta"

    # ------------------------------------------------------------------
    # Identity — auto-incrementing integer PK (simpler than UUID here
    # because BM25 meta rows are few and are queried by institution_id)
    # ------------------------------------------------------------------
    id: int = Column(Integer, primary_key=True, autoincrement=True)

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------
    institution_id: str = Column(String, nullable=False, index=True)

    # ------------------------------------------------------------------
    # Index location & statistics
    # ------------------------------------------------------------------
    index_path: str = Column(String, nullable=False)   # path to serialised file
    document_count: int = Column(Integer, nullable=False, default=0)
    chunk_count: int = Column(Integer, nullable=False, default=0)

    # ------------------------------------------------------------------
    # Build metadata
    # ------------------------------------------------------------------
    built_at: datetime = Column(DateTime, nullable=False, default=_utcnow)
    is_valid: bool = Column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BM25IndexMeta id={self.id} institution={self.institution_id!r} "
            f"chunks={self.chunk_count} valid={self.is_valid}>"
        )
