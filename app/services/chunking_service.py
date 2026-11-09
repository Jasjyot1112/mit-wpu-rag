"""
ChunkingService — structure-aware intelligent text chunking.

Strategies:
- PPTX: Slide-aware (preserve slide_start/slide_end)
- PDF: Page-aware (preserve page_start/page_end)
- DOCX: Section/heading-aware

Configuration:
- chunk_size (tokens, default 800)
- chunk_overlap (tokens, default 120)

Every chunk carries full source metadata for citation accuracy.
"""
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger("rag_engine.services.chunking")

# Approximate chars per token (conservative; actual varies by content)
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A single text chunk with full source metadata."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    institution_id: str = ""
    source_filename: str = ""
    source_type: str = ""  # PDF, PPTX, DOCX
    chunk_index: int = 0
    text: str = ""
    heading: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    slide_start: Optional[int] = None
    slide_end: Optional[int] = None
    section: Optional[str] = None
    extraction_method: str = "digital"
    token_count: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.token_count is None and self.text:
            self.token_count = self._estimate_tokens(self.text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: words / 0.75."""
        words = len(text.split())
        return max(1, int(words / 0.75))


class ChunkingService:
    """Chunk extracted document pages/slides into overlapping text segments."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        institution_id: str = "MIT-WPU",
    ):
        self._chunk_size = chunk_size  # target tokens per chunk
        self._chunk_overlap = chunk_overlap
        self._institution_id = institution_id
        # In chars (approximate)
        self._chunk_chars = chunk_size * _CHARS_PER_TOKEN
        self._overlap_chars = chunk_overlap * _CHARS_PER_TOKEN

    def chunk_extraction_result(
        self, extraction_result, document_id: str
    ) -> List[Chunk]:
        """
        Produce chunks from an ExtractionResult.

        Dispatches to source-type-specific chunkers.
        """
        source_type = extraction_result.source_type
        if source_type == "PPTX":
            return self._chunk_pptx(extraction_result, document_id)
        elif source_type == "PDF":
            return self._chunk_pdf(extraction_result, document_id)
        elif source_type == "DOCX":
            return self._chunk_docx(extraction_result, document_id)
        else:
            logger.warning("Unknown source_type '%s'; using generic chunker.", source_type)
            return self._chunk_generic(extraction_result, document_id)

    # ------------------------------------------------------------------
    # PPTX chunking — slide-aware
    # ------------------------------------------------------------------

    def _chunk_pptx(self, result, document_id: str) -> List[Chunk]:
        """
        For PPTX, prefer keeping each slide as one chunk when it fits.
        When a slide is too long, split it. When consecutive slides are
        very short and topically related (same heading prefix), merge them.
        """
        chunks: List[Chunk] = []
        chunk_index = 0

        # Pass 1: Try to keep each slide whole
        buffer_text = ""
        buffer_start_slide: Optional[int] = None
        buffer_end_slide: Optional[int] = None
        buffer_heading: Optional[str] = None

        for page in result.pages:
            if page.is_empty or not page.text.strip():
                continue

            slide_num = page.slide_number
            text = page.text.strip()
            heading = page.heading

            # If slide text is larger than chunk_size, split independently
            if self._token_count(text) > self._chunk_size:
                # Flush buffer first
                if buffer_text.strip():
                    for chunk in self._split_text_into_chunks(
                        buffer_text, document_id, result.source_filename, "PPTX",
                        chunk_index, page.extraction_method.value,
                        slide_start=buffer_start_slide, slide_end=buffer_end_slide,
                        heading=buffer_heading,
                    ):
                        chunks.append(chunk)
                        chunk_index += 1
                    buffer_text = ""
                    buffer_start_slide = None

                # Split the oversized slide
                for chunk in self._split_text_into_chunks(
                    text, document_id, result.source_filename, "PPTX",
                    chunk_index, page.extraction_method.value,
                    slide_start=slide_num, slide_end=slide_num,
                    heading=heading,
                ):
                    chunks.append(chunk)
                    chunk_index += 1
                continue

            # Try to merge small slides
            combined_tokens = self._token_count(buffer_text + "\n\n" + text)
            if buffer_text and combined_tokens <= self._chunk_size:
                buffer_text = (buffer_text + "\n\n" + text).strip()
                buffer_end_slide = slide_num
                if not buffer_heading and heading:
                    buffer_heading = heading
            else:
                # Flush existing buffer
                if buffer_text.strip():
                    for chunk in self._split_text_into_chunks(
                        buffer_text, document_id, result.source_filename, "PPTX",
                        chunk_index, page.extraction_method.value,
                        slide_start=buffer_start_slide, slide_end=buffer_end_slide,
                        heading=buffer_heading,
                    ):
                        chunks.append(chunk)
                        chunk_index += 1

                # Start new buffer
                buffer_text = text
                buffer_start_slide = slide_num
                buffer_end_slide = slide_num
                buffer_heading = heading

        # Flush final buffer
        if buffer_text.strip():
            for chunk in self._split_text_into_chunks(
                buffer_text, document_id, result.source_filename, "PPTX",
                chunk_index, "slide",
                slide_start=buffer_start_slide, slide_end=buffer_end_slide,
                heading=buffer_heading,
            ):
                chunks.append(chunk)
                chunk_index += 1

        logger.info("PPTX '%s': produced %d chunks from %d slides.",
                    result.source_filename, len(chunks), result.total_slides)
        return self._deduplicate(chunks)

    # ------------------------------------------------------------------
    # PDF chunking — page-aware
    # ------------------------------------------------------------------

    def _chunk_pdf(self, result, document_id: str) -> List[Chunk]:
        """Chunk PDF pages with page boundary awareness."""
        chunks: List[Chunk] = []
        chunk_index = 0
        buffer_text = ""
        buffer_page_start: Optional[int] = None
        buffer_page_end: Optional[int] = None
        buffer_heading: Optional[str] = None
        buffer_method = "digital"

        for page in result.pages:
            if page.is_empty or not page.text.strip():
                continue

            page_num = page.page_number
            text = page.text.strip()

            if self._token_count(text) > self._chunk_size:
                if buffer_text.strip():
                    for chunk in self._split_text_into_chunks(
                        buffer_text, document_id, result.source_filename, "PDF",
                        chunk_index, buffer_method,
                        page_start=buffer_page_start, page_end=buffer_page_end,
                        heading=buffer_heading,
                    ):
                        chunks.append(chunk)
                        chunk_index += 1
                    buffer_text = ""
                    buffer_page_start = None

                for chunk in self._split_text_into_chunks(
                    text, document_id, result.source_filename, "PDF",
                    chunk_index, page.extraction_method.value,
                    page_start=page_num, page_end=page_num,
                    heading=page.heading,
                ):
                    chunks.append(chunk)
                    chunk_index += 1
                continue

            combined_tokens = self._token_count(buffer_text + "\n\n" + text)
            if buffer_text and combined_tokens <= self._chunk_size:
                buffer_text = (buffer_text + "\n\n" + text).strip()
                buffer_page_end = page_num
            else:
                if buffer_text.strip():
                    for chunk in self._split_text_into_chunks(
                        buffer_text, document_id, result.source_filename, "PDF",
                        chunk_index, buffer_method,
                        page_start=buffer_page_start, page_end=buffer_page_end,
                        heading=buffer_heading,
                    ):
                        chunks.append(chunk)
                        chunk_index += 1

                buffer_text = text
                buffer_page_start = page_num
                buffer_page_end = page_num
                buffer_heading = page.heading
                buffer_method = page.extraction_method.value

        if buffer_text.strip():
            for chunk in self._split_text_into_chunks(
                buffer_text, document_id, result.source_filename, "PDF",
                chunk_index, buffer_method,
                page_start=buffer_page_start, page_end=buffer_page_end,
                heading=buffer_heading,
            ):
                chunks.append(chunk)
                chunk_index += 1

        logger.info("PDF '%s': produced %d chunks from %d pages.",
                    result.source_filename, len(chunks), result.total_pages)
        return self._deduplicate(chunks)

    # ------------------------------------------------------------------
    # DOCX chunking — section/heading-aware
    # ------------------------------------------------------------------

    def _chunk_docx(self, result, document_id: str) -> List[Chunk]:
        """Chunk DOCX respecting heading boundaries."""
        chunks: List[Chunk] = []
        chunk_index = 0

        for page in result.pages:
            if page.is_empty or not page.text.strip():
                continue

            text = page.text.strip()
            for chunk in self._split_text_into_chunks(
                text, document_id, result.source_filename, "DOCX",
                chunk_index, page.extraction_method.value,
                heading=page.heading,
                section=page.section,
            ):
                chunks.append(chunk)
                chunk_index += 1

        logger.info("DOCX '%s': produced %d chunks.",
                    result.source_filename, len(chunks))
        return self._deduplicate(chunks)

    def _chunk_generic(self, result, document_id: str) -> List[Chunk]:
        """Generic fallback chunker."""
        all_text = "\n\n".join(p.text for p in result.pages if p.text.strip())
        return self._split_text_into_chunks(
            all_text, document_id, result.source_filename,
            result.source_type, 0, "digital"
        )

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_text_into_chunks(
        self,
        text: str,
        document_id: str,
        source_filename: str,
        source_type: str,
        start_index: int,
        extraction_method: str,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        slide_start: Optional[int] = None,
        slide_end: Optional[int] = None,
        heading: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[Chunk]:
        """Split a text block into overlapping chunks."""
        if not text.strip():
            return []

        # Split into sentences for cleaner boundaries
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_chars = 0
        chunk_local_index = 0

        for sentence in sentences:
            sentence_chars = len(sentence)

            if current_chars + sentence_chars > self._chunk_chars and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(self._make_chunk(
                        chunk_text, document_id, source_filename, source_type,
                        start_index + chunk_local_index, extraction_method,
                        page_start, page_end, slide_start, slide_end,
                        heading, section,
                    ))
                    chunk_local_index += 1

                # Build overlap: keep sentences from end of current until overlap_chars
                overlap_sentences = []
                overlap_chars = 0
                for s in reversed(current_sentences):
                    if overlap_chars + len(s) > self._overlap_chars:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_chars += len(s)

                current_sentences = overlap_sentences
                current_chars = overlap_chars

            current_sentences.append(sentence)
            current_chars += sentence_chars

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(self._make_chunk(
                    chunk_text, document_id, source_filename, source_type,
                    start_index + chunk_local_index, extraction_method,
                    page_start, page_end, slide_start, slide_end,
                    heading, section,
                ))

        return chunks

    def _make_chunk(
        self,
        text: str,
        document_id: str,
        source_filename: str,
        source_type: str,
        chunk_index: int,
        extraction_method: str,
        page_start: Optional[int],
        page_end: Optional[int],
        slide_start: Optional[int],
        slide_end: Optional[int],
        heading: Optional[str],
        section: Optional[str],
    ) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            institution_id=self._institution_id,
            source_filename=source_filename,
            source_type=source_type,
            chunk_index=chunk_index,
            text=text,
            heading=heading,
            page_start=page_start,
            page_end=page_end,
            slide_start=slide_start,
            slide_end=slide_end,
            section=section,
            extraction_method=extraction_method,
        )

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentence-like units.
        Preserves bullet/numbered list items as separate units.
        """
        # Split on paragraph boundaries first
        paragraphs = re.split(r"\n{2,}", text)
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Split paragraph into sentences
            para_sentences = re.split(
                r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)(?=•|\d+\.)", para
            )
            for s in para_sentences:
                s = s.strip()
                if s:
                    sentences.append(s)
        return sentences

    @staticmethod
    def _token_count(text: str) -> int:
        """Estimate token count from text."""
        if not text:
            return 0
        return max(1, int(len(text.split()) / 0.75))

    @staticmethod
    def _deduplicate(chunks: List[Chunk]) -> List[Chunk]:
        """Remove exact-duplicate chunks (same text hash)."""
        seen = set()
        unique = []
        for chunk in chunks:
            text_hash = hashlib.md5(chunk.text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(chunk)
        # Update chunk indices after dedup
        for i, chunk in enumerate(unique):
            chunk.chunk_index = i
        return unique
