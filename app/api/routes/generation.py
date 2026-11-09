"""
AI Generation Router for Grounded Q&A and MCQ Draft Generation.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.security import verify_api_key
from app.models.database import get_db
from app.repositories import chunk_repository
from app.schemas.generation_schemas import (
    AnswerStatus,
    AskRequest,
    AskResponse,
    GeneratedQuestion,
    MCQOption,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
)

router = APIRouter(prefix="/api/v1", tags=["generation"])


def get_hybrid_retrieval(request: Request):
    return request.app.state.hybrid_retrieval_service


def get_reranker(request: Request):
    return request.app.state.reranking_service


def get_context_builder(request: Request):
    return request.app.state.context_builder_service


def get_generator(request: Request):
    return request.app.state.generation_service


def get_question_gen_service(request: Request):
    return request.app.state.question_generation_service


@router.post(
    "/ai/ask",
    response_model=AskResponse,
    summary="Ask a question grounded in authorized educational materials",
)
def ask_question(
    request_data: AskRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    retrieval_service=Depends(get_hybrid_retrieval),
    reranking_service=Depends(get_reranker),
    context_builder=Depends(get_context_builder),
    generator=Depends(get_generator),
):
    """
    Retrieves authorized course materials, reranks candidate passages,
    assembles context, and generates a grounded response with verifiable citations.
    """
    try:
        # 1. Retrieve candidates
        retrieval_res = retrieval_service.retrieve(
            query=request_data.question,
            institution_id=request_data.institution_id,
            document_ids=request_data.document_ids,
            top_k=20,
        )
        candidates = retrieval_res.get("candidates", [])

        # 2. Enrich candidate texts from DB
        missing_ids = [c["chunk_id"] for c in candidates if not c.get("text")]
        if missing_ids:
            chunks_map = {
                c.id: c.text
                for c in db.query(chunk_repository.Chunk)
                .filter(chunk_repository.Chunk.id.in_(missing_ids))
                .all()
            }
            for c in candidates:
                if not c.get("text") and c["chunk_id"] in chunks_map:
                    c["text"] = chunks_map[c["chunk_id"]]

        # 3. Rerank
        reranked, _ = reranking_service.rerank(
            query=request_data.question,
            candidates=candidates,
            top_k=6,
        )

        # 4. Context assembly
        context_str, citations = context_builder.build(
            chunks=reranked,
            query=request_data.question,
            max_chunks=6,
        )

        # 5. Check if we have context
        if not context_str.strip():
            gk_answer = None
            if request_data.include_general_knowledge:
                gk_answer = (
                    "General Knowledge Explanation — Not sourced from uploaded MIT-WPU material:\n\n"
                    + generator.answer_with_general_knowledge(
                        question=request_data.question,
                        student_level=request_data.student_level,
                    )
                )

            return AskResponse(
                status=AnswerStatus.INSUFFICIENT_CONTEXT,
                grounded=False,
                answer=None,
                general_knowledge_answer=gk_answer,
                citations=[],
                conversation_id=request_data.conversation_id,
                insufficient_context_message=(
                    "The uploaded MIT-WPU materials do not contain enough relevant "
                    "information to answer this question."
                ),
                retrieved_chunks_count=len(candidates),
            )

        # 6. Generate answer via Gemini
        res = generator.answer_question(
            question=request_data.question,
            context=context_str,
            student_level=request_data.student_level,
            conversation_history=request_data.conversation_history,
        )

        if res.get("status") == "INSUFFICIENT_CONTEXT":
            gk_answer = None
            if request_data.include_general_knowledge:
                gk_answer = (
                    "General Knowledge Explanation — Not sourced from uploaded MIT-WPU material:\n\n"
                    + generator.answer_with_general_knowledge(
                        question=request_data.question,
                        student_level=request_data.student_level,
                    )
                )

            return AskResponse(
                status=AnswerStatus.INSUFFICIENT_CONTEXT,
                grounded=False,
                answer=None,
                general_knowledge_answer=gk_answer,
                citations=[],
                conversation_id=request_data.conversation_id,
                insufficient_context_message=(
                    "The uploaded MIT-WPU materials do not contain enough relevant "
                    "information to answer this question."
                ),
                retrieved_chunks_count=len(candidates),
            )

        return AskResponse(
            status=AnswerStatus.ANSWERED,
            grounded=True,
            answer=res.get("answer"),
            general_knowledge_answer=None,
            citations=citations,
            conversation_id=request_data.conversation_id,
            insufficient_context_message=None,
            retrieved_chunks_count=len(candidates),
        )

    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/generation/questions",
    response_model=QuestionGenerationResponse,
    summary="Generate grounded MCQ drafts for teacher approval",
)
def generate_questions(
    request_data: QuestionGenerationRequest,
    api_key: str = Depends(verify_api_key),
    question_service=Depends(get_question_gen_service),
):
    """
    Retrieves course materials for subject and topic and produces multiple-choice
    question drafts with answers, explanations, difficulty tags, and citations.
    """
    try:
        res = question_service.generate_mcqs(
            subject=request_data.subject,
            topic=request_data.topic,
            institution_id=request_data.institution_id,
            document_ids=request_data.document_ids,
            easy_count=request_data.easy_count,
            medium_count=request_data.medium_count,
            hard_count=request_data.hard_count,
        )

        formatted_questions = []
        for q in res.get("questions", []):
            options = [MCQOption(**opt) for opt in q.get("options", [])]
            formatted_questions.append(
                GeneratedQuestion(
                    question_text=q.get("question_text", ""),
                    options=options,
                    correct_answer=q.get("correct_answer", "A"),
                    explanation=q.get("explanation", ""),
                    topic=q.get("topic", request_data.topic),
                    difficulty=q.get("difficulty", "medium"),
                    source_document_id=q.get("source_document_id", ""),
                    source_document_name=q.get("source_document_name", ""),
                    source_pages=q.get("source_pages"),
                    source_slides=q.get("source_slides"),
                    supporting_chunk_ids=q.get("supporting_chunk_ids", []),
                )
            )

        return QuestionGenerationResponse(
            subject=res.get("subject", request_data.subject),
            topic=res.get("topic", request_data.topic),
            requested_count=res.get("requested_count", 0),
            generated_count=len(formatted_questions),
            questions=formatted_questions,
            warning=res.get("warning"),
            citations=res.get("citations", []),
        )

    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
