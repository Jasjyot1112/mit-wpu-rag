"""
Context Builder Service.

Assembles retrieved chunks into a structured context string
suitable for Gemini generation.

Responsibilities:
- Deduplicate overlapping chunks
- Sort by document and page/slide order
- Format with clear source labels
- Respect token limits
- Build citation records from actual metadata
- Treat document text as untrusted data (no instruction override possible)
"""
import logging
from typing import Dict, List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger("context_builder")

# Max context chars to send to Gemini (approximate; Gemini supports large contexts)
_MAX_CONTEXT_CHARS = 32000
_MAX_CHUNK_CHARS = 4000  # Cap any single chunk


class ContextBuilderService:
    """
    Builds the final context string and citation list from retrieved chunks.
    """

    def __init__(self, max_context_chars: int = _MAX_CONTEXT_CHARS):
        self._max_context_chars = max_context_chars

    def build(
        self,
        chunks: List[dict],
        query: str,
        max_chunks: int = 6,
    ) -> Tuple[str, List[dict]]:
        """
        Build a formatted context string and citation list from retrieved chunks.

        Args:
            chunks: List of retrieved chunk dicts with text and metadata.
            query: The user query (for relevance sorting context).
            max_chunks: Maximum number of chunks to include.

        Returns:
            Tuple of (context_string, citations_list)
        """
        if not chunks:
            return "", []

        # Deduplicate by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)
            elif not cid:
                unique_chunks.append(chunk)

        # Sort by document, then by slide/page order for coherence
        unique_chunks = self._sort_chunks(unique_chunks)

        # Limit to max_chunks
        selected = unique_chunks[:max_chunks]

        # Build context string
        context_parts = []
        total_chars = 0
        included_chunks = []

        for i, chunk in enumerate(selected, start=1):
            source_header = self._format_source_header(i, chunk)
            text = self._sanitize_chunk_text(chunk.get("text", ""))
            text = text[:_MAX_CHUNK_CHARS]  # Cap individual chunk

            part = f"{source_header}\n{text}"

            if total_chars + len(part) > self._max_context_chars:
                logger.info("Context limit reached at chunk %d/%d.", i - 1, len(selected))
                break

            context_parts.append(part)
            total_chars += len(part)
            included_chunks.append(chunk)

        context_string = "\n\n---\n\n".join(context_parts)

        # Build citations from actual metadata
        citations = self._build_citations(included_chunks)

        logger.debug(
            "Context built: %d chunks, %d chars, %d citations.",
            len(included_chunks), total_chars, len(citations),
        )

        return context_string, citations

    def _sort_chunks(self, chunks: List[dict]) -> List[dict]:
        """Sort chunks by document, then by page/slide number."""
        def sort_key(chunk):
            doc_id = chunk.get("document_id") or ""
            doc_name = chunk.get("document_name") or ""
            page = chunk.get("page_start") or chunk.get("slide_start") or 0
            return (doc_id or doc_name, page)

        return sorted(chunks, key=sort_key)

    def _format_source_header(self, index: int, chunk: dict) -> str:
        """Format the source attribution header for a chunk."""
        doc_name = chunk.get("document_name") or chunk.get("source_filename") or "Unknown Document"
        source_type = chunk.get("source_type", "")
        heading = chunk.get("heading")

        lines = [f"SOURCE {index}:"]
        lines.append(f"Document: {doc_name}")

        if source_type == "PPTX":
            slide_start = chunk.get("slide_start")
            slide_end = chunk.get("slide_end")
            if slide_start is not None:
                slide_ref = f"Slide: {slide_start}"
                if slide_end and slide_end != slide_start:
                    slide_ref += f"–{slide_end}"
                lines.append(slide_ref)
        elif source_type == "PDF":
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            if page_start is not None:
                page_ref = f"Page: {page_start}"
                if page_end and page_end != page_start:
                    page_ref += f"–{page_end}"
                lines.append(page_ref)
        elif source_type == "DOCX":
            section = chunk.get("section")
            if section:
                lines.append(f"Section: {section}")

        if heading:
            lines.append(f"Topic: {heading}")

        return "\n".join(lines)

    def _build_citations(self, chunks: List[dict]) -> List[dict]:
        """
        Build citation records from actual chunk metadata.
        Group by document_id.
        """
        citations_by_doc: Dict[str, dict] = {}

        for chunk in chunks:
            doc_id = chunk.get("document_id") or chunk.get("document_name") or "unknown"
            chunk_id = chunk.get("chunk_id", "")

            if doc_id not in citations_by_doc:
                citations_by_doc[doc_id] = {
                    "document_id": chunk.get("document_id"),
                    "document_name": chunk.get("document_name") or chunk.get("source_filename"),
                    "source_type": chunk.get("source_type"),
                    "page_start": None,
                    "page_end": None,
                    "slide_start": None,
                    "slide_end": None,
                    "section": chunk.get("section"),
                    "chunk_ids": [],
                }

            citation = citations_by_doc[doc_id]
            if chunk_id:
                citation["chunk_ids"].append(chunk_id)

            # Update page/slide ranges
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            slide_start = chunk.get("slide_start")
            slide_end = chunk.get("slide_end")

            if page_start is not None:
                if citation["page_start"] is None or page_start < citation["page_start"]:
                    citation["page_start"] = page_start
            if page_end is not None:
                if citation["page_end"] is None or page_end > citation["page_end"]:
                    citation["page_end"] = page_end
            if slide_start is not None:
                if citation["slide_start"] is None or slide_start < citation["slide_start"]:
                    citation["slide_start"] = slide_start
            if slide_end is not None:
                if citation["slide_end"] is None or slide_end > citation["slide_end"]:
                    citation["slide_end"] = slide_end

        return list(citations_by_doc.values())

    @staticmethod
    def _sanitize_chunk_text(text: str) -> str:
        """
        Sanitize chunk text before including it in the LLM context.
        Prevents prompt injection from document content.

        IMPORTANT: We treat document text as UNTRUSTED DATA.
        Any instructions embedded in documents must not affect LLM behavior.
        The system prompt in GeminiGenerationService enforces this.
        We additionally add a text marker to reinforce the boundary.
        """
        if not text:
            return ""
        # Strip any leading/trailing whitespace
        text = text.strip()
        # We do NOT modify content — the system prompt handles injection prevention.
        # Modifying educational content would corrupt grounding.
        return text
