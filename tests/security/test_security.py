"""
Security and Access Control Tests.
"""

import pytest
from app.core.exceptions import AuthenticationError, UnsupportedFileTypeError, FileSizeLimitError
from app.services.ingestion_service import IngestionService
from app.core.security import verify_api_key
from app.core.config import Settings


class TestSecurityValidation:
    def test_verify_api_key_valid(self):
        settings = Settings(RAG_API_KEY="supersecret123", GEMINI_API_KEY="test", PINECONE_API_KEY="test")
        key = verify_api_key(api_key="supersecret123", settings=settings)
        assert key == "supersecret123"

    def test_verify_api_key_invalid_raises_authentication_error(self):
        settings = Settings(RAG_API_KEY="supersecret123", GEMINI_API_KEY="test", PINECONE_API_KEY="test")
        with pytest.raises(AuthenticationError):
            verify_api_key(api_key="wrong_key", settings=settings)

    def test_verify_api_key_empty_raises_authentication_error(self):
        settings = Settings(RAG_API_KEY="supersecret123", GEMINI_API_KEY="test", PINECONE_API_KEY="test")
        with pytest.raises(AuthenticationError):
            verify_api_key(api_key="", settings=settings)

    def test_unsupported_file_extension_rejected(self):
        settings = Settings(RAG_API_KEY="key", GEMINI_API_KEY="k", PINECONE_API_KEY="p", UPLOAD_DIR="./tmp")
        service = IngestionService(
            db_session_factory=None, pdf_extractor=None, pptx_extractor=None,
            docx_extractor=None, ocr_service=None, text_cleaner=None,
            chunker=None, embedding_service=None, vector_store=None,
            bm25_service=None, settings=settings
        )

        with pytest.raises(UnsupportedFileTypeError):
            service._validate_file("malicious.exe", b"binary content")

        with pytest.raises(UnsupportedFileTypeError):
            service._validate_file("script.sh", b"echo 1")

    def test_file_size_limit_enforced(self):
        settings = Settings(RAG_API_KEY="key", GEMINI_API_KEY="k", PINECONE_API_KEY="p", MAX_FILE_SIZE_MB=1, UPLOAD_DIR="./tmp")
        service = IngestionService(
            db_session_factory=None, pdf_extractor=None, pptx_extractor=None,
            docx_extractor=None, ocr_service=None, text_cleaner=None,
            chunker=None, embedding_service=None, vector_store=None,
            bm25_service=None, settings=settings
        )

        huge_payload = b"A" * (2 * 1024 * 1024)  # 2MB > 1MB limit
        with pytest.raises(FileSizeLimitError):
            service._validate_file("large.pdf", huge_payload)

    def test_path_traversal_sanitized(self):
        malicious_filename = "../../../../../etc/passwd.pptx"
        doc_id = "doc123"
        safe_name = IngestionService._make_safe_filename(malicious_filename, doc_id)
        assert "../" not in safe_name
        assert ".." not in safe_name
        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert safe_name.startswith("doc123_")
