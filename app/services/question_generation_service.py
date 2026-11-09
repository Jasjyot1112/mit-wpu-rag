"""
Question Generation Service — grounded MCQ generation based on retrieved course material.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.exceptions import InsufficientContextError
from app.repositories import chunk_repository

logger = logging.getLogger("rag_engine.services.question_generation")


class QuestionGenerationService:
    """Orchestrates grounded question draft generation for teachers."""

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

    def generate_mcqs(
        self,
        subject: str,
        topic: str,
        institution_id: str = "MIT-WPU",
        document_ids: Optional[List[str]] = None,
        easy_count: int = 3,
        medium_count: int = 4,
        hard_count: int = 3,
    ) -> Dict[str, Any]:
        """Retrieve relevant context and generate grounded MCQs."""
        total_requested = easy_count + medium_count + hard_count
        search_query = f"{subject} {topic} definitions algorithms principles concepts"

        # 1. Retrieve candidates
        retrieval_res = self._retrieval.retrieve(
            query=search_query,
            institution_id=institution_id,
            document_ids=document_ids,
            top_k=25,
        )
        candidates = retrieval_res.get("candidates", [])

        # 2. Enrich candidate texts from DB if needed
        enriched_candidates = self._enrich_candidate_texts(candidates)

        # 3. Rerank
        reranked, _ = self._reranker.rerank(
            query=search_query,
            candidates=enriched_candidates,
            top_k=10,
        )

        # 4. Build context
        context_str, citations = self._context_builder.build(
            chunks=reranked,
            query=search_query,
            max_chunks=8,
        )

        if not context_str.strip():
            return {
                "subject": subject,
                "topic": topic,
                "requested_count": total_requested,
                "generated_count": 0,
                "questions": [],
                "warning": "No relevant educational material found to generate questions.",
                "citations": [],
            }

        # 5. Generate questions via Gemini
        raw_questions = self._generator.generate_questions(
            context=context_str,
            subject=subject,
            topic=topic,
            easy_count=easy_count,
            medium_count=medium_count,
            hard_count=hard_count,
        )

        # 6. Attach citation grounding
        formatted_questions = []
        for q in raw_questions:
            # Map source citation to question
            primary_citation = citations[0] if citations else None
            q_formatted = {
                "question_text": q.get("question_text", ""),
                "options": q.get("options", []),
                "correct_answer": q.get("correct_answer", "A"),
                "explanation": q.get("explanation", ""),
                "topic": topic,
                "difficulty": q.get("difficulty", "medium"),
                "source_document_id": primary_citation["document_id"] if primary_citation else "unknown",
                "source_document_name": primary_citation["document_name"] if primary_citation else "unknown",
                "source_pages": [primary_citation["page_start"]] if primary_citation and primary_citation.get("page_start") else None,
                "source_slides": [primary_citation["slide_start"]] if primary_citation and primary_citation.get("slide_start") else None,
                "supporting_chunk_ids": primary_citation["chunk_ids"] if primary_citation else [],
            }
            formatted_questions.append(q_formatted)

        warning = None
        if len(formatted_questions) < total_requested:
            warning = (
                f"Generated {len(formatted_questions)} questions instead of requested {total_requested}. "
                "The available course material could not reliably support additional ungrounded questions."
            )

        return {
            "subject": subject,
            "topic": topic,
            "requested_count": total_requested,
            "generated_count": len(formatted_questions),
            "questions": formatted_questions,
            "warning": warning,
            "citations": citations,
        }

    def _enrich_candidate_texts(self, candidates: List[dict]) -> List[dict]:
        """Ensure full text is loaded from DB for each retrieved chunk."""
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
