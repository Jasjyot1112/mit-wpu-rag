"""
Unit tests for BM25Service.
"""

import os
import tempfile
import pytest
from app.services.bm25_service import BM25Service


class TestBM25Service:
    def test_build_and_search_technical_terms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "bm25.pkl")
            service = BM25Service(index_path=index_path)

            chunks = [
                {
                    "chunk_id": "c1",
                    "institution_id": "MIT-WPU",
                    "text": "The Bully Election Algorithm elects a coordinator when a crash occurs.",
                    "document_id": "d1",
                    "source_filename": "ds.pptx",
                    "source_type": "PPTX",
                    "slide_start": 18,
                    "slide_end": 18,
                },
                {
                    "chunk_id": "c2",
                    "institution_id": "MIT-WPU",
                    "text": "Lamport Clocks establish logical timestamps for distributed events.",
                    "document_id": "d1",
                    "source_filename": "ds.pptx",
                    "slide_start": 25,
                    "slide_end": 26,
                },
                {
                    "chunk_id": "c3",
                    "institution_id": "MIT-WPU",
                    "text": "TCP is a connection-oriented transmission control protocol with reliable delivery.",
                    "document_id": "d2",
                    "source_filename": "cn.pdf",
                    "page_start": 10,
                    "page_end": 11,
                },
            ]

            service.build_index(chunks, institution_id="MIT-WPU")
            assert service.is_built is True
            assert service.chunk_count == 3

            # Search exact technical query
            results = service.search("Bully Election Algorithm", top_k=2)
            assert len(results) > 0
            assert results[0]["chunk_id"] == "c1"
            assert "Bully" in results[0]["text"]

            # Search second technical query
            results_lamport = service.search("Lamport timestamps", top_k=2)
            assert len(results_lamport) > 0
            assert results_lamport[0]["chunk_id"] == "c2"

    def test_persistence_and_reloading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_bm25.pkl")
            service1 = BM25Service(index_path=index_path)

            chunks = [
                {
                    "chunk_id": "c1",
                    "institution_id": "MIT-WPU",
                    "text": "Two Phase Commit protocol 2PC guarantees atomic transactions.",
                    "document_id": "d1",
                    "source_filename": "db.pptx",
                }
            ]
            service1.build_index(chunks, institution_id="MIT-WPU")

            # Load into new service instance
            service2 = BM25Service(index_path=index_path)
            loaded = service2.load_index()
            assert loaded is True
            assert service2.is_built is True
            assert service2.chunk_count == 1

            results = service2.search("Two Phase Commit", top_k=1)
            assert len(results) == 1
            assert results[0]["chunk_id"] == "c1"

    def test_remove_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_bm25.pkl")
            service = BM25Service(index_path=index_path)

            chunks = [
                {"chunk_id": "c1", "document_id": "doc_A", "text": "AlphaUniqueDocA Content", "institution_id": "MIT-WPU"},
                {"chunk_id": "c2", "document_id": "doc_B", "text": "BetaUniqueDocB Content", "institution_id": "MIT-WPU"},
            ]
            service.build_index(chunks, institution_id="MIT-WPU")
            assert service.chunk_count == 2

            service.remove_document("doc_A")
            assert service.chunk_count == 1
            # doc_A content must not be present
            results_a = service.search("AlphaUniqueDocA")
            assert len(results_a) == 0
            # doc_B content must still be found
            results_b = service.search("BetaUniqueDocB")
            assert len(results_b) == 1
            assert results_b[0]["chunk_id"] == "c2"
