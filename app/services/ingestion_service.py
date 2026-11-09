"""
Ingestion Service — orchestrates document upload, validation, extraction,
chunking, embedding, vector storage, and BM25 index updates.
"""

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.exceptions import (
    DocumentNotFoundError,
    FileSizeLimitError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
)
from app.extractors.base import ExtractionMethod
from app.models.document import Document
from app.repositories import chunk_repository, document_repository

logger = logging.getLogger("rag_engine.services.ingestion")

_ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".docx"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class IngestionService:
    """
    Coordinates ingestion lifecycle for MIT-WPU educational documents.
    """

    def __init__(
        self,
        db_session_factory,
        pdf_extractor,
        pptx_extractor,
        docx_extractor,
        ocr_service,
        text_cleaner,
        chunker,
        embedding_service,
        vector_store,
        bm25_service,
        settings,
    ):
        self._session_factory = db_session_factory
        self._pdf_extractor = pdf_extractor
        self._pptx_extractor = pptx_extractor
        self._docx_extractor = docx_extractor
        self._ocr_service = ocr_service
        self._text_cleaner = text_cleaner
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._bm25_service = bm25_service
        self._settings = settings

        self._upload_dir = Path(settings.UPLOAD_DIR)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Document Upload & Ingestion Job Creation
    # ------------------------------------------------------------------

    def upload_document(
        self,
        db: Session,
        file_content: bytes,
        filename: str,
        institution_id: str,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate, save file to storage, create database record and queue job.
        """
        self._validate_file(filename, file_content)

        checksum = hashlib.sha256(file_content).hexdigest()
        existing_doc = document_repository.get_document_by_checksum(
            db, checksum, institution_id
        )
        if existing_doc and existing_doc.processing_status in ("COMPLETED", "PROCESSING"):
            logger.info(
                "Document '%s' already uploaded (ID: %s, Status: %s)",
                filename, existing_doc.id, existing_doc.processing_status
            )
            # Create a re-ingest job if needed or return existing
            job = document_repository.create_ingestion_job(
                db, existing_doc.id, job_type="REPROCESS"
            )
            return {
                "document_id": existing_doc.id,
                "job_id": job.id,
                "filename": filename,
                "status": existing_doc.processing_status,
                "message": "Existing document found with matching checksum. Requeued for processing.",
            }

        document_id = str(uuid4())
        ext = Path(filename).suffix.lower()
        safe_name = self._make_safe_filename(filename, document_id)
        storage_path = self._upload_dir / safe_name

        with open(storage_path, "wb") as f:
            f.write(file_content)

        doc = document_repository.create_document(
            db=db,
            id=document_id,
            institution_id=institution_id,
            filename=filename,
            safe_filename=safe_name,
            file_extension=ext,
            mime_type=mime_type or self._guess_mime(ext),
            file_size=len(file_content),
            file_path=str(storage_path),
            checksum=checksum,
            processing_status="PENDING",
            indexing_status="PENDING",
            total_chunks=0,
            embedding_model=self._settings.GEMINI_EMBEDDING_MODEL,
        )

        job = document_repository.create_ingestion_job(
            db=db,
            document_id=doc.id,
            job_type="INGEST",
        )

        logger.info(
            "Document '%s' uploaded successfully with ID '%s'. Job ID: '%s'",
            filename, doc.id, job.id
        )

        return {
            "document_id": doc.id,
            "job_id": job.id,
            "filename": filename,
            "status": "PENDING",
            "message": "Document uploaded successfully and queued for ingestion.",
        }

    # ------------------------------------------------------------------
    # Background Processing (Called by Worker)
    # ------------------------------------------------------------------

    def process_document(self, document_id: str) -> None:
        """
        Complete ingestion pipeline executed by background worker.
        """
        db = self._session_factory()
        try:
            doc = document_repository.get_document(db, document_id)
            if not doc:
                logger.error("process_document: Document '%s' not found.", document_id)
                return

            logger.info("Processing document '%s' (%s)...", doc.filename, document_id)
            document_repository.update_document_status(
                db,
                document_id,
                status="PROCESSING",
                processing_started_at=_now_utc(),
                error_message=None,
            )

            file_path = doc.file_path
            if not os.path.exists(file_path):
                raise InvalidDocumentError(f"Stored file not found on disk at '{file_path}'")

            ext = doc.file_extension.lower()
            extractor = self._get_extractor(ext)

            # Step 1: Extraction
            extraction_result = extractor.extract(
                file_path=file_path,
                document_id=document_id,
                filename=doc.filename,
            )

            # Record non-fatal extraction errors
            for err in extraction_result.errors:
                page_or_slide = err.get("page_or_slide")
                page_num = page_or_slide if ext == ".pdf" else None
                slide_num = page_or_slide if ext == ".pptx" else None
                document_repository.create_extraction_error(
                    db=db,
                    document_id=document_id,
                    page_number=page_num,
                    slide_number=slide_num,
                    error_type=err.get("error_type", "EXTRACTION_ERROR"),
                    error_message=err.get("message", "Unknown extraction error"),
                )

            # Step 2: Clean extracted pages
            for page in extraction_result.pages:
                if page.text:
                    page.text = self._text_cleaner.clean(
                        page.text, source_type=extraction_result.source_type
                    )

            # Step 3: Chunking
            chunks = self._chunker.chunk_extraction_result(
                extraction_result, document_id=document_id
            )

            if not chunks:
                logger.warning("No text chunks generated for document '%s'", doc.filename)
                document_repository.update_document_status(
                    db,
                    document_id,
                    status="COMPLETED",
                    processing_completed_at=_now_utc(),
                    total_pages=extraction_result.total_pages,
                    total_slides=extraction_result.total_slides,
                    total_chunks=0,
                    error_message="Document processed but yielded no extractable text content.",
                )
                return

            # Step 4: Persist chunks to database
            chunk_records = []
            for c in chunks:
                chunk_records.append({
                    "id": c.chunk_id,
                    "document_id": document_id,
                    "institution_id": doc.institution_id,
                    "source_filename": doc.filename,
                    "source_type": c.source_type,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "heading": c.heading,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "slide_start": c.slide_start,
                    "slide_end": c.slide_end,
                    "section": c.section,
                    "extraction_method": c.extraction_method,
                    "token_count": c.token_count,
                    "is_indexed": False,
                })

            created_chunks = chunk_repository.create_chunks(db, chunk_records)
            logger.info("Persisted %d chunks to database.", len(created_chunks))

            # Step 5: Generate Gemini Embeddings
            chunk_texts = [c.text for c in created_chunks]
            embeddings = self._embedding_service.embed_documents(chunk_texts)

            if len(embeddings) != len(created_chunks):
                raise ValueError("Mismatch between chunk count and generated embedding count.")

            # Step 6: Upsert to Pinecone Vector Store
            pinecone_records = []
            chunk_ids = []
            vector_ids = []

            for chunk_obj, embedding in zip(created_chunks, embeddings):
                vector_id = f"{document_id}_{chunk_obj.id}"
                metadata = {
                    "institution_id": doc.institution_id,
                    "document_id": document_id,
                    "chunk_id": chunk_obj.id,
                    "source_filename": doc.filename,
                    "source_type": chunk_obj.source_type,
                    "chunk_index": chunk_obj.chunk_index,
                    "heading": chunk_obj.heading or "",
                    "page_start": chunk_obj.page_start,
                    "page_end": chunk_obj.page_end,
                    "slide_start": chunk_obj.slide_start,
                    "slide_end": chunk_obj.slide_end,
                    "section": chunk_obj.section or "",
                    "extraction_method": chunk_obj.extraction_method,
                    # Truncate text preview for Pinecone metadata limit
                    "text_preview": chunk_obj.text[:300],
                }
                pinecone_records.append((vector_id, embedding, metadata))
                chunk_ids.append(chunk_obj.id)
                vector_ids.append(vector_id)

            self._vector_store.upsert_vectors(pinecone_records)
            chunk_repository.mark_chunks_indexed(db, chunk_ids, vector_ids)
            logger.info("Indexed %d vectors in Pinecone.", len(pinecone_records))

            # Step 7: Update BM25 Index
            all_inst_chunks = chunk_repository.get_all_chunks_for_bm25(
                db, institution_id=doc.institution_id
            )
            self._bm25_service.build_index(all_inst_chunks, institution_id=doc.institution_id)
            logger.info("Updated BM25 index with %d total chunks.", len(all_inst_chunks))

            # Step 8: Complete document processing
            primary_method = extraction_result.primary_extraction_method
            method_str = primary_method.value if hasattr(primary_method, "value") else str(primary_method)

            document_repository.update_document_processing_complete(
                db=db,
                document_id=document_id,
                total_chunks=len(created_chunks),
                extraction_method=method_str,
                embedding_model=self._settings.GEMINI_EMBEDDING_MODEL,
            )
            # Update pages/slides count
            document_repository.update_document_status(
                db,
                document_id,
                status="COMPLETED",
                total_pages=extraction_result.total_pages,
                total_slides=extraction_result.total_slides,
            )

            logger.info("Successfully finished processing document '%s'.", doc.filename)

        except Exception as exc:
            logger.error("Processing failed for document '%s': %s", document_id, exc, exc_info=True)
            document_repository.update_document_status(
                db,
                document_id,
                status="FAILED",
                error_message=str(exc),
                processing_completed_at=_now_utc(),
            )
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Reprocess & Delete
    # ------------------------------------------------------------------

    def reprocess_document(self, db: Session, document_id: str) -> Dict[str, Any]:
        """Clean old chunks/vectors and trigger re-ingestion job."""
        doc = document_repository.get_document(db, document_id)
        if not doc:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")

        # Clean old vectors from Pinecone
        try:
            self._vector_store.delete_vectors_by_document(document_id)
        except Exception as exc:
            logger.warning("Pinecone vector deletion warning during reprocess: %s", exc)

        # Clean old chunks from SQLite
        chunk_repository.delete_chunks_by_document(db, document_id)

        # Remove from BM25 index
        self._bm25_service.remove_document(document_id)

        # Reset document status and create REPROCESS job
        document_repository.update_document_status(
            db,
            document_id,
            status="PENDING",
            indexing_status="PENDING",
            total_chunks=0,
            error_message=None,
        )

        job = document_repository.create_ingestion_job(
            db, document_id=document_id, job_type="REPROCESS"
        )

        logger.info("Document '%s' queued for reprocessing. Job ID: '%s'", doc.filename, job.id)
        return {
            "document_id": document_id,
            "job_id": job.id,
            "filename": doc.filename,
            "status": "PENDING",
            "message": "Document scheduled for reprocessing.",
        }

    def delete_document(self, db: Session, document_id: str) -> bool:
        """Delete document, its chunks, its vectors, and stored file."""
        doc = document_repository.get_document(db, document_id)
        if not doc:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")

        # Delete vectors from Pinecone
        try:
            self._vector_store.delete_vectors_by_document(document_id)
        except Exception as exc:
            logger.warning("Pinecone deletion warning: %s", exc)

        # Remove from BM25 index
        self._bm25_service.remove_document(document_id)

        # Delete chunks from SQLite
        chunk_repository.delete_chunks_by_document(db, document_id)

        # Delete physical file from disk
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except OSError as exc:
                logger.warning("Could not delete physical file '%s': %s", doc.file_path, exc)

        # Delete Document record
        success = document_repository.delete_document(db, document_id)
        logger.info("Deleted document '%s' (ID: %s)", doc.filename, document_id)
        return success

    # ------------------------------------------------------------------
    # Validation and Utilities
    # ------------------------------------------------------------------

    def _validate_file(self, filename: str, content: bytes) -> None:
        """Validate filename, extension, and file size."""
        if not filename or not filename.strip():
            raise InvalidDocumentError("Filename cannot be empty.")

        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File type '{ext}' is unsupported. Allowed formats: {', '.join(_ALLOWED_EXTENSIONS)}"
            )

        max_bytes = self._settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise FileSizeLimitError(
                f"File size {len(content)} bytes exceeds maximum configured limit of "
                f"{self._settings.MAX_FILE_SIZE_MB} MB."
            )

        if len(content) == 0:
            raise InvalidDocumentError("Uploaded file is empty (0 bytes).")

    def _get_extractor(self, ext: str):
        """Return the appropriate extractor for the file extension."""
        if ext == ".pdf":
            return self._pdf_extractor
        elif ext in (".pptx", ".ppt"):
            return self._pptx_extractor
        elif ext in (".docx", ".doc"):
            return self._docx_extractor
        raise UnsupportedFileTypeError(f"No extractor available for extension '{ext}'")

    @staticmethod
    def _make_safe_filename(filename: str, doc_id: str) -> str:
        """Sanitize filename to prevent path traversal and collisions."""
        base_name = Path(filename).name
        stem = Path(base_name).stem
        ext = Path(base_name).suffix
        safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")
        safe_ext = re.sub(r"[^a-zA-Z0-9.]", "", ext).lower()
        if not safe_stem:
            safe_stem = "document"
        return f"{doc_id}_{safe_stem[:60]}{safe_ext}"

    @staticmethod
    def _guess_mime(ext: str) -> str:
        mimes = {
            ".pdf": "application/pdf",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }
        return mimes.get(ext.lower(), "application/octet-stream")
