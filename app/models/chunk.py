"""
SQLAlchemy model for text chunks extracted from documents.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


def _utcnow() -> datetime:
    """Return current UTC time (timezone-naive for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Chunk(Base):
    """
    A single text chunk extracted from a :class:`Document`.

    Chunks are the atomic unit of retrieval in the RAG pipeline.  Each
    chunk is embedded independently and stored in the vector index with
    ``vector_id`` pointing to the corresponding Pinecone record.
    """

    __tablename__ = "chunks"

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
    institution_id: str = Column(String, nullable=False, index=True)

    # ------------------------------------------------------------------
    # Source metadata
    # ------------------------------------------------------------------
    source_filename: str = Column(String, nullable=False)
    source_type: str = Column(String, nullable=False)   # PDF | PPTX | DOCX

    # ------------------------------------------------------------------
    # Chunk content & position
    # ------------------------------------------------------------------
    chunk_index: int = Column(Integer, nullable=False)   # 0-based position
    text: str = Column(Text, nullable=False)             # actual chunk text

    # ------------------------------------------------------------------
    # Optional structural context
    # ------------------------------------------------------------------
    heading: str | None = Column(String, nullable=True)
    page_start: int | None = Column(Integer, nullable=True)
    page_end: int | None = Column(Integer, nullable=True)
    slide_start: int | None = Column(Integer, nullable=True)
    slide_end: int | None = Column(Integer, nullable=True)
    section: str | None = Column(String, nullable=True)

    # ------------------------------------------------------------------
    # Extraction quality
    # extraction_method values: digital | ocr | mixed
    # ------------------------------------------------------------------
    extraction_method: str = Column(String, nullable=False, default="digital")
    token_count: int | None = Column(Integer, nullable=True)

    # ------------------------------------------------------------------
    # Vector index linkage
    # ------------------------------------------------------------------
    is_indexed: bool = Column(Boolean, nullable=False, default=False)
    vector_id: str | None = Column(String, nullable=True)  # Pinecone vector ID

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
        back_populates="chunks",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Chunk id={self.id!r} document_id={self.document_id!r} "
            f"index={self.chunk_index} indexed={self.is_indexed}>"
        )


# ---------------------------------------------------------------------------
# Composite index — fast lookup of ordered chunks within a document
# ---------------------------------------------------------------------------
Index(
    "ix_chunks_document_id_chunk_index",
    Chunk.document_id,
    Chunk.chunk_index,
)
