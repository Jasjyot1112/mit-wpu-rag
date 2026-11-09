"""
AI Tutor Router for Interactive Doubt Solving.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.core.security import verify_api_key
from app.schemas.generation_schemas import AnswerStatus, TutorRequest, TutorResponse

router = APIRouter(prefix="/api/v1/ai", tags=["tutor"])


def get_tutor_service(request: Request):
    return request.app.state.tutor_service


@router.post(
    "/tutor",
    response_model=TutorResponse,
    summary="Interactive AI Tutor concept explanation and doubt solver",
)
def tutor_doubt_solver(
    request_data: TutorRequest,
    api_key: str = Depends(verify_api_key),
    tutor_service=Depends(get_tutor_service),
):
    try:
        res = tutor_service.explain(
            question=request_data.question,
            institution_id=request_data.institution_id,
            subject=request_data.subject,
            topic=request_data.topic,
            document_ids=request_data.document_ids,
            conversation_id=request_data.conversation_id,
            conversation_history=request_data.conversation_history,
            student_level=request_data.student_level,
            request_examples=request_data.request_examples,
        )

        return TutorResponse(
            status=AnswerStatus(res.get("status", "ANSWERED")),
            grounded=res.get("grounded", False),
            explanation=res.get("explanation"),
            examples=res.get("examples"),
            key_concepts=res.get("key_concepts"),
            citations=res.get("citations", []),
            conversation_id=res.get("conversation_id"),
            insufficient_context_message=res.get("insufficient_context_message"),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
