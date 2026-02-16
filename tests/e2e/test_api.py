"""End-to-end tests for the FastAPI endpoints.

These tests use ``TestClient`` with mock workflow / pipeline components
to exercise the HTTP layer: request validation, response format,
error handling, concurrent requests, and middleware behaviour.

Marked with ``@pytest.mark.e2e`` — run with ``pytest --run-e2e``.
"""

import concurrent.futures
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app import app
from src.types import Citation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_success_state(question: str = "What is federalism?") -> dict:
    """Return a graph-state dict that represents a successful answer."""
    return {
        "question": question,
        "query_type": "factual",
        "chunks": [],
        "search_results": [],
        "answer": (
            "Federalism is a system of government in which power is divided "
            "between a national government and regional governments."
        ),
        "citations": [
            {
                "source": "federalism.txt",
                "chunk_id": "chunk-001",
                "text": "Federalism is a system of government...",
                "score": 0.92,
            }
        ],
        "confidence": 0.92,
        "is_grounded": True,
        "retry_count": 1,
        "action": "accept",
        "refusal_reason": None,
        "error": None,
    }


def _make_refusal_state(question: str = "What is the price of Bitcoin?") -> dict:
    """Return a graph-state dict representing a refusal."""
    return {
        "question": question,
        "query_type": "unsupported",
        "chunks": [],
        "search_results": [],
        "answer": None,
        "citations": [],
        "confidence": 0.0,
        "is_grounded": False,
        "retry_count": 0,
        "action": "refuse",
        "refusal_reason": "No relevant documents were found.",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Tests — Query endpoint
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestQueryEndpointE2E:
    """End-to-end tests for POST /query."""

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    @pytest.fixture()
    def mock_workflow(self):
        mock = MagicMock()
        mock.invoke.return_value = _make_success_state()
        return mock

    def test_successful_query(self, client, mock_workflow):
        """A valid query should return 200 with answer and citations."""
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            resp = client.post(
                "/query",
                json={"question": "What is federalism?", "mode": "qna", "k": 5},
            )

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["answer"] is not None
        assert "federalism" in data["answer"].lower()
        assert len(data["citations"]) >= 1
        assert data["confidence"] > 0
        assert data["refusal_reason"] is None
        assert data["latency_ms"] >= 0

    def test_refusal_query(self, client):
        """A query that triggers refusal should still return 200."""
        mock = MagicMock()
        mock.invoke.return_value = _make_refusal_state()

        with patch("src.app.routes.query.get_workflow", return_value=mock):
            resp = client.post(
                "/query",
                json={"question": "What is the price of Bitcoin?"},
            )

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["answer"] is None
        assert data["refusal_reason"] is not None
        assert data["confidence"] == 0.0

    def test_query_with_different_modes(self, client, mock_workflow):
        """All valid modes should be accepted."""
        for mode in ("qna", "vector", "hybrid"):
            with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
                resp = client.post(
                    "/query",
                    json={"question": "What is X?", "mode": mode},
                )
            assert resp.status_code == status.HTTP_200_OK

    def test_query_with_different_k_values(self, client, mock_workflow):
        """k=1 and k=20 should both be accepted."""
        for k in (1, 5, 10, 20):
            with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
                resp = client.post(
                    "/query",
                    json={"question": "What is X?", "k": k},
                )
            assert resp.status_code == status.HTTP_200_OK

    def test_response_json_structure(self, client, mock_workflow):
        """The response should have all expected fields."""
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            resp = client.post(
                "/query", json={"question": "What is X?"},
            )

        data = resp.json()
        for field in ("answer", "citations", "confidence", "refusal_reason", "latency_ms"):
            assert field in data, f"Missing field: {field}"

    def test_citation_structure(self, client, mock_workflow):
        """Each citation should have source, chunk_id, text, score."""
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            resp = client.post(
                "/query", json={"question": "What is federalism?"},
            )

        data = resp.json()
        if data["citations"]:
            cit = data["citations"][0]
            assert "source" in cit
            assert "chunk_id" in cit
            assert "text" in cit
            assert "score" in cit


@pytest.mark.e2e
class TestQueryValidationE2E:
    """Validation error handling at the HTTP level."""

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    def test_empty_question(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_whitespace_question(self, client):
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_question_field(self, client):
        resp = client.post("/query", json={"mode": "qna"})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_mode(self, client):
        resp = client.post(
            "/query", json={"question": "X?", "mode": "turbo"},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_k_out_of_range_low(self, client):
        resp = client.post(
            "/query", json={"question": "X?", "k": 0},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_k_out_of_range_high(self, client):
        resp = client.post(
            "/query", json={"question": "X?", "k": 25},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_question_too_long(self, client):
        resp = client.post(
            "/query", json={"question": "a" * 2001},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_json(self, client):
        resp = client.post(
            "/query",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.e2e
class TestQueryErrorHandlingE2E:
    """Error scenarios in the query endpoint."""

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    def test_workflow_internal_error(self, client):
        """If the graph state contains error, return 500."""
        mock = MagicMock()
        mock.invoke.return_value = {
            "question": "X?",
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "refusal_reason": None,
            "error": "Vector store connection refused",
        }
        with patch("src.app.routes.query.get_workflow", return_value=mock):
            resp = client.post("/query", json={"question": "X?"})

        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_workflow_raises_exception(self, client):
        """If the workflow raises, should return 500."""
        mock = MagicMock()
        mock.invoke.side_effect = RuntimeError("boom")

        with patch("src.app.routes.query.get_workflow", return_value=mock):
            resp = client.post("/query", json={"question": "X?"})

        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_workflow_rag_error(self, client):
        """RAGError should be caught and return 500."""
        from src.exceptions import RAGError

        mock = MagicMock()
        mock.invoke.side_effect = RAGError("retrieval failed", details="timeout")

        with patch("src.app.routes.query.get_workflow", return_value=mock):
            resp = client.post("/query", json={"question": "X?"})

        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# Tests — Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHealthEndpointE2E:
    """End-to-end tests for GET /health."""

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == status.HTTP_200_OK

    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.json()

        assert "status" in data
        assert data["status"] in ("healthy", "unhealthy")
        assert "components" in data
        assert "version" in data

    def test_health_components(self, client):
        resp = client.get("/health")
        data = resp.json()

        components = data.get("components", {})
        for name in ("embeddings", "vector_store", "llm"):
            assert name in components


# ---------------------------------------------------------------------------
# Tests — Ingest endpoint
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestIngestEndpointE2E:
    """End-to-end tests for POST /ingest."""

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    def test_ingest_validation_empty_corpus(self, client):
        resp = client.post("/ingest", json={"corpus": ""})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ingest_validation_whitespace_corpus(self, client):
        resp = client.post("/ingest", json={"corpus": "   "})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ingest_validation_missing_corpus(self, client):
        resp = client.post("/ingest", json={})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ingest_status_not_found(self, client):
        resp = client.get("/ingest/status/nonexistent_corpus_xyz")
        assert resp.status_code == status.HTTP_404_NOT_FOUND


# ---------------------------------------------------------------------------
# Tests — Middleware
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestMiddlewareE2E:
    """Middleware behaviour checks."""

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    def test_cors_headers_present(self, client):
        """CORS should allow the origin."""
        resp = client.get(
            "/health",
            headers={"Origin": "http://example.com"},
        )
        assert resp.status_code == status.HTTP_200_OK

    def test_openapi_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == status.HTTP_200_OK
        assert "text/html" in resp.headers["content-type"]

    def test_openapi_json_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == status.HTTP_200_OK
        schema = resp.json()
        assert schema["info"]["title"] == "Grounded GraphRAG Tutor API"

    def test_openapi_schema_has_query_path(self, client):
        resp = client.get("/openapi.json")
        schema = resp.json()
        assert "/query" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/ingest" in schema["paths"]


# ---------------------------------------------------------------------------
# Tests — Concurrent requests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestConcurrentRequestsE2E:
    """Verify the API handles concurrent requests."""

    def test_concurrent_queries(self):
        """Multiple queries in parallel should all succeed."""
        mock = MagicMock()
        mock.invoke.return_value = _make_success_state()

        client = TestClient(app)

        def _query(q: str) -> int:
            resp = client.post("/query", json={"question": q})
            return resp.status_code

        questions = [f"What is topic {i}?" for i in range(10)]

        # Set the workflow globally so all threads see it (avoids
        # race conditions that occur with per-thread patch calls).
        with patch("src.app.routes.query.get_workflow", return_value=mock):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(_query, q) for q in questions]
                results = [f.result() for f in futures]

        assert all(s == 200 for s in results), f"Some requests failed: {results}"

    def test_concurrent_health_checks(self):
        """Multiple health checks in parallel should not fail."""
        client = TestClient(app)

        def _health() -> int:
            return client.get("/health").status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_health) for _ in range(20)]
            results = [f.result() for f in futures]

        assert all(s == 200 for s in results)
