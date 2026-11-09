"""
SQLAlchemy synchronous engine setup for the MIT-WPU RAG Engine.
Uses sync SQLAlchemy (not async) — simpler for SQLite + background worker.
"""

from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------------------
# Base — all models inherit from this
# ---------------------------------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------------------------------
# Module-level references (populated after create_db_engine is called)
# ---------------------------------------------------------------------------
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def create_db_engine(database_url: str) -> Engine:
    """
    Create and return a SQLAlchemy engine.

    For SQLite URLs the ``check_same_thread=False`` connect arg is added
    automatically so the engine can safely be used from background threads.
    ``pool_pre_ping=True`` ensures stale connections are recycled.

    Args:
        database_url: SQLAlchemy-compatible DB URL, e.g.
            ``"sqlite:///./rag.db"`` or
            ``"postgresql+psycopg2://user:pass@host/db"``.

    Returns:
        A configured :class:`sqlalchemy.engine.Engine` instance.
    """
    global _engine, _SessionLocal

    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine,
    )

    return _engine


def SessionLocal() -> Session:
    """Return a new Session instance from the sessionmaker."""
    if _SessionLocal is None:
        raise RuntimeError(
            "Database engine has not been initialised. "
            "Call create_db_engine() before using SessionLocal()."
        )
    return _SessionLocal()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures it is
    closed after the request completes (even on error).

    Usage::

        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    if _SessionLocal is None:
        raise RuntimeError(
            "Database engine has not been initialised. "
            "Call create_db_engine() before using get_db()."
        )
    db: Session = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(engine: Engine) -> None:
    """
    Create all tables that are registered on ``Base.metadata``.

    This should be called once at application startup *after* all model
    modules have been imported so that every model is registered on Base.

    Args:
        engine: The engine returned by :func:`create_db_engine`.
    """
    Base.metadata.create_all(bind=engine)
