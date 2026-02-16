"""Tests for the query endpoint.

This module tests the POST /query endpoint including:
- Valid request handling
- Invalid request validation
- Error handling
- Response format validation
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.app import app
from src.app.schemas import QueryRequest, QueryResponse
from src.types import Citation
from src.exceptions import RAGError, LLMError


class TestQueryRequestValidation:
    """Tests for QueryRequest schema validation."""
    
    def test_valid_request(self):
        """Test that a valid request passes validation."""
        request = QueryRequest(
            question="What is federalism?",
            mode="qna",
            k=5
        )
        assert request.question == "What is federalism?"
        assert request.mode == "qna"
        assert request.k == 5
    
    def test_valid_request_with_defaults(self):
        """Test that defaults are applied correctly."""
        request = QueryRequest(question="What is federalism?")
        assert request.mode == "qna"
        assert request.k == 5
    
    def test_invalid_request_empty_question(self):
        """Test that empty question fails validation."""
        with pytest.raises(ValueError):
            QueryRequest(question="")
    
    def test_invalid_request_whitespace_question(self):
        """Test that whitespace-only question fails validation."""
        with pytest.raises(ValueError):
            QueryRequest(question="   ")
    
    def test_invalid_request_too_long_question(self):
        """Test that question over 2000 chars fails validation."""
        long_question = "a" * 2001
        with pytest.raises(ValueError):
            QueryRequest(question=long_question)
    
    def test_invalid_request_invalid_mode(self):
        """Test that invalid mode fails validation."""
        with pytest.raises(ValueError):
            QueryRequest(question="What?", mode="invalid")
    
    def test_invalid_request_k_too_small(self):
        """Test that k < 1 fails validation."""
        with pytest.raises(ValueError):
            QueryRequest(question="What?", k=0)
    
    def test_invalid_request_k_too_large(self):
        """Test that k > 20 fails validation."""
        with pytest.raises(ValueError):
            QueryRequest(question="What?", k=21)
    
    def test_valid_request_all_modes(self):
        """Test that all valid modes are accepted."""
        for mode in ["qna", "vector", "hybrid"]:
            request = QueryRequest(question="What?", mode=mode)
            assert request.mode == mode


class TestQueryResponseFormat:
    """Tests for QueryResponse schema format."""
    
    def test_response_with_answer(self):
        """Test response with a valid answer."""
        citation = Citation(
            source="test.txt",
            chunk_id="chunk-1",
            text="Some text",
            score=0.95
        )
        response = QueryResponse(
            answer="Federalism is a system of government...",
            citations=[citation],
            confidence=0.92,
            refusal_reason=None,
            latency_ms=1234.5
        )
        assert response.answer == "Federalism is a system of government..."
        assert len(response.citations) == 1
        assert response.confidence == 0.92
        assert response.refusal_reason is None
        assert response.latency_ms == 1234.5
    
    def test_response_with_refusal(self):
        """Test response with a refusal reason."""
        response = QueryResponse(
            answer=None,
            citations=[],
            confidence=0.0,
            refusal_reason="Unable to answer due to insufficient context",
            latency_ms=500.0
        )
        assert response.answer is None
        assert response.refusal_reason == "Unable to answer due to insufficient context"
    
    def test_response_defaults(self):
        """Test response with default values."""
        response = QueryResponse()
        assert response.answer is None
        assert response.citations == []
        assert response.confidence == 0.0
        assert response.refusal_reason is None
        assert response.latency_ms == 0.0


class TestQueryEndpoint:
    """Tests for the POST /query endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow that returns a successful result."""
        mock = MagicMock()
        mock.invoke.return_value = {
            "question": "What is federalism?",
            "query_type": "factual",
            "chunks": [],
            "search_results": [],
            "answer": "Federalism is a system of government where power is divided.",
            "citations": [
                {
                    "source": "civics.txt",
                    "chunk_id": "chunk-1",
                    "text": "Federalism divides power...",
                    "score": 0.95,
                }
            ],
            "confidence": 0.92,
            "is_grounded": True,
            "retry_count": 0,
            "action": "accept",
            "refusal_reason": None,
            "error": None,
        }
        return mock
    
    def test_query_endpoint_success(self, client, mock_workflow):
        """Test successful query processing."""
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["answer"] == "Federalism is a system of government where power is divided."
        assert data["confidence"] == 0.92
        assert data["refusal_reason"] is None
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0
    
    def test_query_endpoint_with_refusal(self, client, mock_workflow):
        """Test query processing with refusal."""
        mock_workflow.invoke.return_value = {
            "question": "What is your opinion on politics?",
            "query_type": "unsupported",
            "chunks": [],
            "search_results": [],
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "is_grounded": False,
            "retry_count": 0,
            "action": "refuse",
            "refusal_reason": "This question type is not supported",
            "error": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is your opinion on politics?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["answer"] is None
        assert data["refusal_reason"] == "This question type is not supported"
    
    def test_query_endpoint_with_citations(self, client, mock_workflow):
        """Test query response includes citations."""
        mock_workflow.invoke.return_value = {
            "question": "What is federalism?",
            "query_type": "factual",
            "chunks": [],
            "search_results": [],
            "answer": "Federalism is a system of government.",
            "citations": [
                {
                    "source": "civics.txt",
                    "chunk_id": "chunk-1",
                    "text": "Federalism divides power between national and state governments.",
                    "score": 0.95,
                },
                {
                    "source": "constitution.txt",
                    "chunk_id": "chunk-5",
                    "text": "The Tenth Amendment reserves powers to the states.",
                    "score": 0.88,
                }
            ],
            "confidence": 0.90,
            "is_grounded": True,
            "retry_count": 0,
            "action": "accept",
            "refusal_reason": None,
            "error": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["citations"]) == 2
        assert data["citations"][0]["source"] == "civics.txt"
        assert data["citations"][0]["score"] == 0.95


class TestQueryEndpointValidation:
    """Tests for request validation at the endpoint level."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_missing_question(self, client):
        """Test that missing question returns 422."""
        response = client.post(
            "/query",
            json={
                "mode": "qna",
                "k": 5
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_empty_question(self, client):
        """Test that empty question returns 422."""
        response = client.post(
            "/query",
            json={
                "question": "",
                "mode": "qna",
                "k": 5
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_mode(self, client):
        """Test that invalid mode returns 422."""
        response = client.post(
            "/query",
            json={
                "question": "What is federalism?",
                "mode": "invalid_mode",
                "k": 5
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_k_value(self, client):
        """Test that invalid k value returns 422."""
        response = client.post(
            "/query",
            json={
                "question": "What is federalism?",
                "mode": "qna",
                "k": 0
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_k_exceeds_maximum(self, client):
        """Test that k > 20 returns 422."""
        response = client.post(
            "/query",
            json={
                "question": "What is federalism?",
                "mode": "qna",
                "k": 25
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestQueryEndpointErrorHandling:
    """Tests for error handling in the query endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_workflow_error(self, client):
        """Test handling of workflow errors."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "question": "What is federalism?",
            "error": "Connection refused to vector store",
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "refusal_reason": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_rag_error(self, client):
        """Test handling of RAGError exceptions."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.side_effect = RAGError(
            "Retrieval failed",
            details="Vector store connection error"
        )
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_unexpected_error(self, client):
        """Test handling of unexpected exceptions."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.side_effect = RuntimeError("Unexpected error")
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestQueryEndpointLatency:
    """Tests for latency measurement in the query endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_latency_is_measured(self, client):
        """Test that latency is measured and returned."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "question": "What is federalism?",
            "answer": "Federalism is a system of government.",
            "citations": [],
            "confidence": 0.9,
            "refusal_reason": None,
            "error": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0


class TestQueryEndpointModes:
    """Tests for different query modes."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_qna_mode(self, client):
        """Test qna mode is accepted."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "question": "What is federalism?",
            "answer": "Federalism is a system of government.",
            "citations": [],
            "confidence": 0.9,
            "refusal_reason": None,
            "error": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "qna",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_vector_mode(self, client):
        """Test vector mode is accepted."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "question": "What is federalism?",
            "answer": "Federalism is a system of government.",
            "citations": [],
            "confidence": 0.9,
            "refusal_reason": None,
            "error": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "vector",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_hybrid_mode(self, client):
        """Test hybrid mode is accepted."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "question": "What is federalism?",
            "answer": "Federalism is a system of government.",
            "citations": [],
            "confidence": 0.9,
            "refusal_reason": None,
            "error": None,
        }
        
        with patch("src.app.routes.query.get_workflow", return_value=mock_workflow):
            response = client.post(
                "/query",
                json={
                    "question": "What is federalism?",
                    "mode": "hybrid",
                    "k": 5
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
