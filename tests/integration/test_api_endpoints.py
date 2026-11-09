"""
Integration tests for FastAPI endpoints.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Set testing environment variables before importing app
os.environ["RAG_API_KEY"] = "test_secret_key"
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["PINECONE_API_KEY"] = "test_pinecone_key"
os.environ["DATABASE_URL"] = "sqlite:///./test_rag.db"
os.environ["RERANKER_ENABLED"] = "false"

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["institution_id"] == "MIT-WPU"
    assert "service" in data


def test_unauthorized_request_rejected(client):
    response = client.get("/api/v1/documents")
    # Missing X-API-Key should return 401
    assert response.status_code == 401


def test_authorized_documents_list_empty(client):
    headers = {"X-API-Key": "test_secret_key"}
    response = client.get("/api/v1/documents", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
    assert "total" in data


def test_upload_invalid_file_extension(client):
    headers = {"X-API-Key": "test_secret_key"}
    files = {"file": ("malicious.exe", b"test content", "application/octet-stream")}
    response = client.post("/api/v1/documents/upload", headers=headers, files=files)
    assert response.status_code in (400, 415)
