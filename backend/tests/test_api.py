"""FastAPI integration tests using TestClient verifying status codes, auth, and shapes."""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings

client = TestClient(app)


def test_health_endpoint_public():
    """Verify /health is accessible without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "llm_provider" in data
    assert "embedding_provider" in data


def test_auth_unauthorized_on_missing_key():
    """Verify /query and /ingest reject requests missing X-API-Key with 401."""
    # Test query endpoint
    res_query = client.post("/query", json={"query": "Hello"})
    assert res_query.status_code == 401
    assert "Missing required 'X-API-Key' header" in res_query.json()["detail"]

    # Test ingest endpoint
    res_ingest = client.post("/ingest", files={"file": ("test.txt", b"sample content", "text/plain")})
    assert res_ingest.status_code == 401


def test_auth_unauthorized_on_invalid_key():
    """Verify /query rejects invalid API keys."""
    headers = {"X-API-Key": "wrong-invalid-key"}
    response = client.post("/query", json={"query": "Hello"}, headers=headers)
    assert response.status_code == 401
    assert "Invalid 'X-API-Key' provided" in response.json()["detail"]


def test_ingest_and_query_flow():
    """Test file upload via /ingest followed by /query execution with valid API key."""
    settings = get_settings()
    headers = {"X-API-Key": settings.API_KEY}

    # 1. Ingest document
    file_content = b"# Acme SLA\nAcme offers a 99.95% uptime guarantee for Enterprise tier accounts."
    file_obj = io.BytesIO(file_content)

    ingest_res = client.post(
        "/ingest",
        files={"file": ("sla_test.md", file_obj, "text/markdown")},
        headers=headers,
    )
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["chunk_count"] >= 1
    assert ingest_data["filename"] == "sla_test.md"

    # 2. List documents
    docs_res = client.get("/documents")
    assert docs_res.status_code == 200
    docs = docs_res.json()
    assert any(d["filename"] == "sla_test.md" for d in docs)

    # 3. Query with valid key
    query_payload = {
        "query": "What is the uptime guarantee for Enterprise tier?",
        "conversation_id": "test-session-123",
    }
    query_res = client.post("/query", json=query_payload, headers=headers)
    assert query_res.status_code == 200
    res_data = query_res.json()

    # Validate response shape
    assert "answer" in res_data
    assert "citations" in res_data
    assert "confidence" in res_data
    assert "can_answer" in res_data
    assert res_data["conversation_id"] == "test-session-123"
    assert "latency_ms" in res_data
    assert isinstance(res_data["citations"], list)
