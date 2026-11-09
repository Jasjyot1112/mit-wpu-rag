"""
AI Tutor Service — grounded doubt-solving and educational tutoring.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.repositories import chunk_repository

logger = logging.getLogger("rag_engine.services.tutor")


class TutorService:
    """Provides grounded answers, examples, and concept explanations to students."""

    def __init__(
        self,
        hybrid_retrieval_service,
        reranking_service,
        context_builder_service,
        generation_service,
        db_session_factory,
    ):
        self._retrieval = hybrid_retrieval_service
        self._reranker = reranking_service
        self._context_builder = context_builder_service
        self._generator = generation_service
        self._session_factory = db_session_factory

    def explain(
        self,
        question: str,
        institution_id: str = "MIT-WPU",
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
        student_level: str = "undergraduate",
        request_examples: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve authoritative course material and generate an explanation."""
        query_text = question
        if topic:
            query_text = f"{topic} {question}"

        # 1. Retrieve candidates
        retrieval_res = self._retrieval.retrieve(
            query=query_text,
            institution_id=institution_id,
            document_ids=document_ids,
            top_k=20,
        )
        candidates = retrieval_res.get("candidates", [])

        # 2. Enrich candidate texts from DB
        enriched = self._enrich_candidate_texts(candidates)

        # 3. Rerank
        reranked, _ = self._reranker.rerank(
            query=query_text,
            candidates=enriched,
            top_k=6,
        )

        # 4. Build context
        context_str, citations = self._context_builder.build(
            chunks=reranked,
            query=query_text,
            max_chunks=6,
        )

        if not context_str.strip():
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "grounded": False,
                "explanation": None,
                "examples": None,
                "key_concepts": None,
                "citations": [],
                "conversation_id": conversation_id,
                "insufficient_context_message": (
                    "The uploaded MIT-WPU course materials do not contain sufficient "
                    "information to answer this question."
                ),
            }

        # 5. Generate explanation via Gemini
        res = self._generator.tutor_explain(
            question=question,
            context=context_str,
            subject=subject,
            topic=topic,
            student_level=student_level,
            request_examples=request_examples,
        )

        if res.get("is_insufficient"):
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "grounded": False,
                "explanation": None,
                "examples": None,
                "key_concepts": None,
                "citations": [],
                "conversation_id": conversation_id,
                "insufficient_context_message": (
                    "The uploaded MIT-WPU course materials do not contain sufficient "
                    "information to answer this question."
                ),
            }

        return {
            "status": "ANSWERED",
            "grounded": True,
            "explanation": res.get("explanation"),
            "examples": None,
            "key_concepts": None,
            "citations": citations,
            "conversation_id": conversation_id,
            "insufficient_context_message": None,
        }

    def _enrich_candidate_texts(self, candidates: List[dict]) -> List[dict]:
        missing_ids = [c["chunk_id"] for c in candidates if not c.get("text")]
        if not missing_ids:
            return candidates

        db: Session = self._session_factory()
        try:
            chunks_map = {
                c.id: c.text
                for c in db.query(chunk_repository.Chunk)
                .filter(chunk_repository.Chunk.id.in_(missing_ids))
                .all()
            }
            for c in candidates:
                if not c.get("text") and c["chunk_id"] in chunks_map:
                    c["text"] = chunks_map[c["chunk_id"]]
            return candidates
        finally:
            db.close()
