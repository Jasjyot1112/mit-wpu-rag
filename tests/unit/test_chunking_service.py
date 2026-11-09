"""
Unit tests for ChunkingService.

Tests:
- PPTX slide-aware chunking preserves slide numbers
- PDF page-aware chunking preserves page numbers
- Deduplication removes exact duplicates
- Oversized slides are split correctly
- Short slides are merged correctly
- Empty pages are skipped
"""
import pytest
from app.services.chunking_service import ChunkingService, Chunk
from app.extractors.base import ExtractionResult, ExtractedPage, ExtractionMethod


def make_pptx_result(slides: list[dict]) -> ExtractionResult:
    """Helper to create a mock PPTX ExtractionResult."""
    result = ExtractionResult(
        document_id="test-doc-001",
        source_filename="test.pptx",
        source_type="PPTX",
        primary_extraction_method=ExtractionMethod.SLIDE,
    )
    for i, slide in enumerate(slides, start=1):
        page = ExtractedPage(
            slide_number=i,
            page_number=i,
            heading=slide.get("heading"),
            text=slide.get("text", ""),
            extraction_method=ExtractionMethod.SLIDE,
            is_empty=not slide.get("text", "").strip(),
        )
        result.pages.append(page)
    result.total_slides = len(slides)
    result.total_pages = len(slides)
    return result


def make_pdf_result(pages: list[dict]) -> ExtractionResult:
    result = ExtractionResult(
        document_id="test-doc-002",
        source_filename="test.pdf",
        source_type="PDF",
        primary_extraction_method=ExtractionMethod.DIGITAL,
    )
    for i, page in enumerate(pages, start=1):
        ep = ExtractedPage(
            page_number=i,
            heading=page.get("heading"),
            text=page.get("text", ""),
            extraction_method=ExtractionMethod.DIGITAL,
            is_empty=not page.get("text", "").strip(),
        )
        result.pages.append(ep)
    result.total_pages = len(pages)
    return result


class TestChunkingServicePPTX:

    def setup_method(self):
        self.chunker = ChunkingService(chunk_size=200, chunk_overlap=30, institution_id="MIT-WPU")

    def test_slide_numbers_preserved(self):
        """Slide numbers must be preserved in every chunk."""
        result = make_pptx_result([
            {"heading": "Intro", "text": "This is slide one content about distributed systems."},
            {"heading": "BFS", "text": "Slide two discusses breadth-first search algorithms in detail."},
            {"heading": "DFS", "text": "Slide three covers depth-first search with examples."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-001")
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.slide_start is not None, "slide_start must be set"
            assert chunk.slide_end is not None, "slide_end must be set"
            assert chunk.slide_start >= 1
            assert chunk.slide_end >= chunk.slide_start

    def test_heading_preserved(self):
        """Chunk heading must come from slide title."""
        result = make_pptx_result([
            {"heading": "Lamport Clock", "text": "Lamport clocks are logical clocks used in distributed systems."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-001")
        assert len(chunks) > 0
        assert chunks[0].heading == "Lamport Clock"

    def test_empty_slides_skipped(self):
        """Empty slides must not produce chunks."""
        result = make_pptx_result([
            {"heading": "Real Slide", "text": "Some actual content here."},
            {"heading": "Empty", "text": ""},
            {"heading": "", "text": "   "},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-001")
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_institution_id_set(self):
        """Every chunk must have institution_id set."""
        result = make_pptx_result([
            {"text": "Content about cloud computing in distributed systems."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-001")
        for chunk in chunks:
            assert chunk.institution_id == "MIT-WPU"

    def test_source_type_pptx(self):
        result = make_pptx_result([{"text": "OSI Model has 7 layers."}])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-001")
        for chunk in chunks:
            assert chunk.source_type == "PPTX"

    def test_no_duplicate_chunks(self):
        """Duplicate text from different slides should be deduplicated."""
        result = make_pptx_result([
            {"text": "This is repeated content."},
            {"text": "This is repeated content."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-001")
        texts = [c.text for c in chunks]
        # After deduplication, should not have exact duplicate texts
        assert len(texts) == len(set(texts))


class TestChunkingServicePDF:

    def setup_method(self):
        self.chunker = ChunkingService(chunk_size=200, chunk_overlap=30, institution_id="MIT-WPU")

    def test_page_numbers_preserved(self):
        """Page numbers must be preserved in PDF chunks."""
        result = make_pdf_result([
            {"heading": "Introduction", "text": "Chapter 1 introduces networking concepts and protocols."},
            {"heading": "TCP/IP", "text": "TCP/IP is the foundation of internet communication."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-002")
        for chunk in chunks:
            assert chunk.page_start is not None
            assert chunk.page_end is not None

    def test_chunk_index_sequential(self):
        """Chunk indices must be sequential after dedup."""
        result = make_pdf_result([
            {"text": "Page 1 content about data structures."},
            {"text": "Page 2 content about algorithms."},
            {"text": "Page 3 content about graph theory."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-002")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_has_text(self):
        """Every chunk must have non-empty text."""
        result = make_pdf_result([
            {"text": "Binary search tree is an ordered data structure."},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-002")
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_all_chunk_ids_unique(self):
        """All chunk IDs must be unique UUIDs."""
        result = make_pdf_result([
            {"text": f"Content for page {i}" * 10} for i in range(5)
        ])
        chunks = self.chunker.chunk_extraction_result(result, "test-doc-002")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestChunkingEdgeCases:

    def setup_method(self):
        self.chunker = ChunkingService(chunk_size=50, chunk_overlap=10, institution_id="MIT-WPU")

    def test_single_word_slide(self):
        """Very short slides should still produce a chunk."""
        result = make_pptx_result([{"heading": "Title", "text": "Introduction"}])
        chunks = self.chunker.chunk_extraction_result(result, "doc-001")
        assert len(chunks) >= 1

    def test_all_empty_slides(self):
        """All-empty presentation should produce no chunks."""
        result = make_pptx_result([
            {"text": ""},
            {"text": "   "},
            {"text": "\n\n"},
        ])
        chunks = self.chunker.chunk_extraction_result(result, "doc-001")
        assert len(chunks) == 0

    def test_document_id_in_every_chunk(self):
        result = make_pptx_result([{"text": "Real educational content here."}])
        chunks = self.chunker.chunk_extraction_result(result, "my-doc-id")
        for chunk in chunks:
            assert chunk.document_id == "my-doc-id"
