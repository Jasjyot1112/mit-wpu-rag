"""
Document Management Router.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.core.security import verify_api_key, get_current_institution
from app.models.database import get_db
from app.repositories import chunk_repository, document_repository
from app.schemas.document_schemas import (
    ChunksListResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_ingestion_service(request: Request):
    """Retrieve IngestionService singleton from application state."""
    return request.app.state.ingestion_service


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and queue educational document for ingestion",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, PPTX, or DOCX)"),
    institution_id: str = Form(default="MIT-WPU", description="Institutional namespace"),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    service=Depends(get_ingestion_service),
):
    """
    Accepts educational course material, stores the file safely,
    creates metadata records, and queues the background ingestion job.
    """
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    try:
        res = service.upload_document(
            db=db,
            file_content=content,
            filename=file.filename or "unnamed_document",
            institution_id=institution_id,
            mime_type=file.content_type,
        )
        return DocumentUploadResponse(
            id=res["document_id"],
            filename=res["filename"],
            processing_status=res["status"],
            job_id=res["job_id"],
            message=res["message"],
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List uploaded documents for institution",
)
def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    institution_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    current_inst: str = Depends(get_current_institution),
):
    inst = institution_id or current_inst
    docs = document_repository.list_documents(db, institution_id=inst, skip=skip, limit=limit)
    total = document_repository.count_documents(db, institution_id=inst)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details by ID",
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    doc = document_repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return DocumentResponse.model_validate(doc)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Check ingestion and indexing status of a document",
)
def get_document_status(
    document_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    doc = document_repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return DocumentStatusResponse.model_validate(doc)


@router.get(
    "/{document_id}/chunks",
    response_model=ChunksListResponse,
    summary="List extracted chunks for a document",
)
def get_document_chunks(
    document_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    doc = document_repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    chunks = chunk_repository.get_chunks_by_document(db, document_id, skip=skip, limit=limit)
    total = chunk_repository.count_chunks(db, document_id)
    return ChunksListResponse(
        chunks=chunks,
        total=total,
        document_id=document_id,
    )


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger reprocessing of an existing document",
)
def reprocess_document(
    document_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    service=Depends(get_ingestion_service),
):
    try:
        res = service.reprocess_document(db, document_id)
        return DocumentUploadResponse(
            id=res["document_id"],
            filename=res["filename"],
            processing_status=res["status"],
            job_id=res["job_id"],
            message=res["message"],
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete(
    "/{document_id}",
    summary="Delete a document and purge vectors from Pinecone",
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    service=Depends(get_ingestion_service),
):
    success = service.delete_document(db, document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return {
        "status": "DELETED",
        "document_id": document_id,
        "message": "Document, chunks, and vector store entries deleted successfully.",
    }
