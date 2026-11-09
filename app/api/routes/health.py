"""
Health Check and System Diagnostics Router.
"""

from fastapi import APIRouter, Depends
from app.core.config import Settings, get_settings

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Return health status and high-level service metadata."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "institution_id": settings.INSTITUTION_ID,
        "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "generation_model": settings.GEMINI_GENERATION_MODEL,
        "vector_database": "Pinecone",
        "pinecone_index": settings.PINECONE_INDEX_NAME,
        "reranker_enabled": settings.RERANKER_ENABLED,
        "reranker_model": settings.RERANKER_MODEL if settings.RERANKER_ENABLED else None,
    }
