"""
Unit tests for ContextBuilderService.
"""

import pytest
from app.services.context_builder_service import ContextBuilderService


class TestContextBuilderService:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.builder = ContextBuilderService(max_context_chars=10000)

    def test_builds_clear_source_headers_and_citations(self):
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "document_name": "Distributed Systems Unit 2.pptx",
                "source_type": "PPTX",
                "slide_start": 18,
                "slide_end": 18,
                "heading": "Bully Election Algorithm",
                "text": "The Bully Algorithm elect a coordinator process with highest identifier.",
            },
            {
                "chunk_id": "c2",
                "document_id": "doc1",
                "document_name": "Distributed Systems Unit 2.pptx",
                "source_type": "PPTX",
                "slide_start": 19,
                "slide_end": 19,
                "heading": "Bully Election Complexity",
                "text": "Message complexity in worst case is O(N^2) where N is number of processes.",
            },
        ]

        context, citations = self.builder.build(chunks, query="Bully Election Algorithm")

        assert "SOURCE 1:" in context
        assert "Distributed Systems Unit 2.pptx" in context
        assert "Slide: 18" in context
        assert "Bully Election Algorithm" in context
        assert "SOURCE 2:" in context
        assert "Slide: 19" in context

        # Citations verification
        assert len(citations) == 1
        cit = citations[0]
        assert cit["document_id"] == "doc1"
        assert cit["document_name"] == "Distributed Systems Unit 2.pptx"
        assert cit["slide_start"] == 18
        assert cit["slide_end"] == 19
        assert "c1" in cit["chunk_ids"]
        assert "c2" in cit["chunk_ids"]

    def test_deduplicates_chunks(self):
        chunks = [
            {"chunk_id": "dup", "document_id": "d1", "document_name": "doc.pdf", "source_type": "PDF", "text": "Same text"},
            {"chunk_id": "dup", "document_id": "d1", "document_name": "doc.pdf", "source_type": "PDF", "text": "Same text"},
        ]
        context, citations = self.builder.build(chunks, query="test")
        assert "SOURCE 2:" not in context
        assert len(citations[0]["chunk_ids"]) == 1
