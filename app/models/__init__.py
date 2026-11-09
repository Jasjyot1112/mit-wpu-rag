"""
Public API for the ``app.models`` package.

Import order matters: ``database`` must be imported first so that ``Base``
is available before model modules import it.  All model classes are then
registered on ``Base.metadata`` as a side-effect of their import.
"""

from .database import Base, create_db_engine, get_db, init_db, SessionLocal
from .document import Document
from .chunk import Chunk
from .ingestion_job import IngestionJob
from .extraction_error import ExtractionError
from .bm25_index_meta import BM25IndexMeta

__all__ = [
    "Base",
    "get_db",
    "init_db",
    "create_db_engine",
    "SessionLocal",
    "Document",
    "Chunk",
    "IngestionJob",
    "ExtractionError",
    "BM25IndexMeta",
]
