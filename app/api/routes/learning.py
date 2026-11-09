"""
Personalized Learning Router for Summaries, Flashcards, and Revision Notes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.core.security import verify_api_key
from app.schemas.learning_schemas import (
    Flashcard,
    FlashcardsRequest,
    FlashcardsResponse,
    RevisionNotesRequest,
    RevisionNotesResponse,
    RevisionSection,
    SummaryRequest,
    SummaryResponse,
)

router = APIRouter(prefix="/api/v1/learning", tags=["learning"])


def get_learning_service(request: Request):
    return request.app.state.learning_service


@router.post(
    "/summary",
    response_model=SummaryResponse,
    summary="Generate grounded topic summary with key points",
)
def generate_summary(
    request_data: SummaryRequest,
    api_key: str = Depends(verify_api_key),
    learning_service=Depends(get_learning_service),
):
    try:
        res = learning_service.get_summary(
            institution_id=request_data.institution_id,
            topic=request_data.topic,
            subject=request_data.subject,
            document_ids=request_data.document_ids,
            detail_level=request_data.detail_level,
        )
        return SummaryResponse(
            topic=res["topic"],
            summary=res["summary"],
            key_points=res["key_points"],
            citations=res["citations"],
            grounded=res["grounded"],
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/flashcards",
    response_model=FlashcardsResponse,
    summary="Generate revision flashcards grounded in course materials",
)
def generate_flashcards(
    request_data: FlashcardsRequest,
    api_key: str = Depends(verify_api_key),
    learning_service=Depends(get_learning_service),
):
    try:
        res = learning_service.get_flashcards(
            institution_id=request_data.institution_id,
            topic=request_data.topic,
            subject=request_data.subject,
            document_ids=request_data.document_ids,
            count=request_data.count,
        )
        flashcards = [Flashcard(**f) for f in res["flashcards"]]
        return FlashcardsResponse(
            topic=res["topic"],
            flashcards=flashcards,
            generated_count=res["generated_count"],
            citations=res["citations"],
            grounded=res["grounded"],
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/revision-notes",
    response_model=RevisionNotesResponse,
    summary="Generate structured revision notes with definitions and examples",
)
def generate_revision_notes(
    request_data: RevisionNotesRequest,
    api_key: str = Depends(verify_api_key),
    learning_service=Depends(get_learning_service),
):
    try:
        res = learning_service.get_revision_notes(
            institution_id=request_data.institution_id,
            topic=request_data.topic,
            subject=request_data.subject,
            document_ids=request_data.document_ids,
        )
        sections = [RevisionSection(**sec) for sec in res["sections"]]
        return RevisionNotesResponse(
            topic=res["topic"],
            sections=sections,
            citations=res["citations"],
            grounded=res["grounded"],
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
