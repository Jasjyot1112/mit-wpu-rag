"""
MIT-WPU RAG Engine — Main Application Entrypoint.

Sets up FastAPI, database, services, background workers, and REST routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, generation, health, learning, retrieval, tutor
from app.core.config import get_settings
from app.core.exceptions import RAGBaseException
from app.core.logging import get_logger, setup_logging
from app.extractors.docx_extractor import DOCXExtractor
from app.extractors.pdf_extractor import PDFExtractor
from app.extractors.pptx_extractor import PPTXExtractor
from app.integrations.gemini.embeddings import EmbeddingService
from app.integrations.gemini.generation import GeminiGenerationService
from app.integrations.pinecone.vector_store import PineconeVectorStore
from app.models.database import SessionLocal, create_db_engine, init_db
from app.repositories import chunk_repository
from app.services.bm25_service import BM25Service
from app.services.chunking_service import ChunkingService
from app.services.context_builder_service import ContextBuilderService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.ingestion_service import IngestionService
from app.services.learning_service import LearningService
from app.services.ocr_service import OCRService
from app.services.question_generation_service import QuestionGenerationService
from app.services.reranking_service import RerankingService
from app.services.text_cleaning_service import TextCleaningService
from app.services.tutor_service import TutorService
from app.workers.ingestion_worker import IngestionWorker

settings = get_settings()
setup_logging(debug=settings.DEBUG)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and graceful shutdown."""
    logger.info("Initializing %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    # 1. Database initialization
    engine = create_db_engine(settings.DATABASE_URL)
    init_db(engine)
    logger.info("Database schema initialized.")

    # 2. Extractors & OCR
    ocr_service = OCRService(
        tesseract_cmd=settings.TESSERACT_CMD,
        language=settings.OCR_LANGUAGE,
        dpi=settings.OCR_DPI,
    )
    pdf_extractor = PDFExtractor(ocr_service=ocr_service, max_pages=settings.MAX_PDF_PAGES)
    pptx_extractor = PPTXExtractor(ocr_service=ocr_service)
    docx_extractor = DOCXExtractor()

    # 3. NLP & Chunking
    text_cleaner = TextCleaningService()
    chunker = ChunkingService(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        institution_id=settings.INSTITUTION_ID,
    )

    # 4. External Integrations (Gemini & Pinecone)
    embedding_service = None
    if settings.GEMINI_API_KEY:
        try:
            embedding_service = EmbeddingService(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_EMBEDDING_MODEL,
                dimension=settings.EMBEDDING_DIMENSION,
                batch_size=settings.GEMINI_BATCH_SIZE,
                max_retries=settings.GEMINI_MAX_RETRIES,
                task_type_document=settings.GEMINI_EMBEDDING_TASK_TYPE,
                task_type_query=settings.GEMINI_QUERY_TASK_TYPE,
                timeout=settings.GEMINI_TIMEOUT,
            )
            logger.info("Embedding service initialized.")
        except Exception as exc:
            logger.warning("Could not initialize EmbeddingService: %s", exc)

    vector_store = None
    if settings.PINECONE_API_KEY:
        try:
            vector_store = PineconeVectorStore(
                api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME,
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
                dimension=settings.EMBEDDING_DIMENSION,
                namespace=settings.PINECONE_NAMESPACE,
            )
            # Initialize index if online
            vector_store.initialize()
            logger.info("Pinecone vector store initialized.")
        except Exception as exc:
            logger.warning("Could not initialize PineconeVectorStore: %s", exc)

    # 5. BM25 Keyword Search
    bm25_service = BM25Service()
    if not bm25_service.load_index():
        logger.info("Rebuilding BM25 index from stored chunks...")
        db = SessionLocal()
        try:
            stored_chunks = chunk_repository.get_all_chunks_for_bm25(
                db, institution_id=settings.INSTITUTION_ID
            )
            if stored_chunks:
                bm25_service.build_index(stored_chunks, institution_id=settings.INSTITUTION_ID)
        finally:
            db.close()

    # 6. Reranking Service
    reranker = RerankingService(
        model_name=settings.RERANKER_MODEL,
        enabled=settings.RERANKER_ENABLED,
        max_candidates=settings.RERANKER_MAX_CANDIDATES,
    )
    if settings.RERANKER_ENABLED:
        try:
            reranker.load_model()
        except Exception as exc:
            logger.warning("Reranker pre-load warning: %s", exc)

    # 7. Context Builder & LLM Generation
    context_builder = ContextBuilderService()
    generation_service = None
    if settings.GEMINI_API_KEY:
        try:
            generation_service = GeminiGenerationService(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_GENERATION_MODEL,
                max_retries=settings.GEMINI_MAX_RETRIES,
                timeout=settings.GEMINI_TIMEOUT,
            )
            logger.info("Gemini generation service initialized.")
        except Exception as exc:
            logger.warning("Could not initialize GeminiGenerationService: %s", exc)

    # 8. High-Level Composite Services
    hybrid_retrieval = HybridRetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        dense_top_k=settings.DENSE_TOP_K,
        bm25_top_k=settings.BM25_TOP_K,
        fusion_top_k=settings.FUSION_TOP_K,
        final_top_k=settings.FINAL_CONTEXT_TOP_K,
        rrf_k=settings.RRF_K,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
    )

    question_service = QuestionGenerationService(
        hybrid_retrieval_service=hybrid_retrieval,
        reranking_service=reranker,
        context_builder_service=context_builder,
        generation_service=generation_service,
        db_session_factory=SessionLocal,
    )

    tutor_service = TutorService(
        hybrid_retrieval_service=hybrid_retrieval,
        reranking_service=reranker,
        context_builder_service=context_builder,
        generation_service=generation_service,
        db_session_factory=SessionLocal,
    )

    learning_service = LearningService(
        hybrid_retrieval_service=hybrid_retrieval,
        reranking_service=reranker,
        context_builder_service=context_builder,
        generation_service=generation_service,
        db_session_factory=SessionLocal,
    )

    ingestion_service = IngestionService(
        db_session_factory=SessionLocal,
        pdf_extractor=pdf_extractor,
        pptx_extractor=pptx_extractor,
        docx_extractor=docx_extractor,
        ocr_service=ocr_service,
        text_cleaner=text_cleaner,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        settings=settings,
    )

    # 9. Background Ingestion Worker
    worker = IngestionWorker(
        ingestion_service=ingestion_service,
        db_session_factory=SessionLocal,
        max_concurrency=settings.MAX_WORKER_CONCURRENCY,
        job_retry_limit=settings.JOB_RETRY_LIMIT,
    )
    worker.start()

    # Store state on application object
    app.state.engine = engine
    app.state.ocr_service = ocr_service
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.bm25_service = bm25_service
    app.state.reranking_service = reranker
    app.state.context_builder_service = context_builder
    app.state.generation_service = generation_service
    app.state.hybrid_retrieval_service = hybrid_retrieval
    app.state.question_generation_service = question_service
    app.state.tutor_service = tutor_service
    app.state.learning_service = learning_service
    app.state.ingestion_service = ingestion_service
    app.state.worker = worker

    logger.info("MIT-WPU RAG Engine startup sequence complete.")

    yield

    # Shutdown
    logger.info("Initiating graceful shutdown...")
    worker.stop()
    engine.dispose()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="End-to-End Grounded Retrieval-Augmented Generation Engine for MIT-WPU.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    application.include_router(health.router)
    application.include_router(documents.router)
    application.include_router(retrieval.router)
    application.include_router(generation.router)
    application.include_router(tutor.router)
    application.include_router(learning.router)

    # Global Exception Handlers
    @application.exception_handler(RAGBaseException)
    async def rag_exception_handler(request: Request, exc: RAGBaseException):
        logger.error("RAG Error [%s]: %s", exc.__class__.__name__, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "status_code": exc.status_code,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation Error: %s", exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Invalid request parameters.",
                "details": exc.errors(),
            },
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled Exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An internal system error occurred.",
            },
        )

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
