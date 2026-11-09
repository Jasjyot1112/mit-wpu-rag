"""
Personalized Learning Service — grounded summaries, flashcards, and revision notes.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.repositories import chunk_repository

logger = logging.getLogger("rag_engine.services.learning")


class LearningService:
    """Produces grounded revision aids from retrieved course material."""

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

    def get_summary(
        self,
        institution_id: str,
        topic: str,
        subject: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        detail_level: str = "medium",
    ) -> Dict[str, Any]:
        """Generate a grounded topic summary."""
        query_text = f"{subject or ''} {topic} overview concepts summary".strip()
        context_str, citations = self._retrieve_and_build_context(
            query=query_text,
            institution_id=institution_id,
            document_ids=document_ids,
            top_k=6,
        )

        if not context_str.strip():
            return {
                "topic": topic,
                "summary": "No relevant course material found to summarize this topic.",
                "key_points": [],
                "citations": [],
                "grounded": False,
            }

        res = self._generator.generate_summary(
            context=context_str, topic=topic, detail_level=detail_level
        )

        if res.get("is_insufficient"):
            return {
                "topic": topic,
                "summary": "Insufficient information in uploaded course materials to summarize this topic.",
                "key_points": [],
                "citations": [],
                "grounded": False,
            }

        return {
            "topic": topic,
            "summary": res.get("summary", ""),
            "key_points": res.get("key_points", []),
            "citations": citations,
            "grounded": True,
        }

    def get_flashcards(
        self,
        institution_id: str,
        topic: str,
        subject: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        count: int = 10,
    ) -> Dict[str, Any]:
        """Generate flashcards from authorized course materials."""
        query_text = f"{subject or ''} {topic} definitions key terms concepts".strip()
        context_str, citations = self._retrieve_and_build_context(
            query=query_text,
            institution_id=institution_id,
            document_ids=document_ids,
            top_k=8,
        )

        if not context_str.strip():
            return {
                "topic": topic,
                "flashcards": [],
                "generated_count": 0,
                "citations": [],
                "grounded": False,
            }

        res = self._generator.generate_flashcards(
            context=context_str, topic=topic, count=count
        )

        raw_cards = res.get("flashcards", [])
        primary_citation = citations[0] if citations else None
        formatted_cards = []

        for card in raw_cards:
            formatted_cards.append({
                "front": card.get("front", ""),
                "back": card.get("back", ""),
                "topic": topic,
                "difficulty": card.get("difficulty", "medium"),
                "source_document_name": primary_citation["document_name"] if primary_citation else "Course Material",
                "page_start": primary_citation.get("page_start") if primary_citation else None,
                "page_end": primary_citation.get("page_end") if primary_citation else None,
                "slide_start": primary_citation.get("slide_start") if primary_citation else None,
                "slide_end": primary_citation.get("slide_end") if primary_citation else None,
            })

        return {
            "topic": topic,
            "flashcards": formatted_cards,
            "generated_count": len(formatted_cards),
            "citations": citations,
            "grounded": len(formatted_cards) > 0,
        }

    def get_revision_notes(
        self,
        institution_id: str,
        topic: str,
        subject: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate structured revision notes."""
        query_text = f"{subject or ''} {topic} complete guide review revision notes".strip()
        context_str, citations = self._retrieve_and_build_context(
            query=query_text,
            institution_id=institution_id,
            document_ids=document_ids,
            top_k=8,
        )

        if not context_str.strip():
            return {
                "topic": topic,
                "sections": [],
                "citations": [],
                "grounded": False,
            }

        res = self._generator.generate_revision_notes(
            context=context_str, topic=topic
        )

        if res.get("is_insufficient"):
            return {
                "topic": topic,
                "sections": [],
                "citations": [],
                "grounded": False,
            }

        return {
            "topic": topic,
            "sections": res.get("sections", []),
            "citations": citations,
            "grounded": True,
        }

    def _retrieve_and_build_context(
        self, query: str, institution_id: str, document_ids: Optional[List[str]], top_k: int
    ):
        retrieval_res = self._retrieval.retrieve(
            query=query,
            institution_id=institution_id,
            document_ids=document_ids,
            top_k=20,
        )
        candidates = retrieval_res.get("candidates", [])
        enriched = self._enrich_candidate_texts(candidates)
        reranked, _ = self._reranker.rerank(query=query, candidates=enriched, top_k=top_k)
        return self._context_builder.build(chunks=reranked, query=query, max_chunks=top_k)

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
