"""Tests for the ingest endpoint.

This module tests the POST /ingest endpoint including:
- Valid request handling
- Invalid request validation
- Error handling
- Response format validation
- Async ingestion support
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.app import app
from src.app.schemas import IngestRequest, IngestResponse
from src.types import IngestResult
from src.exceptions import RAGError, ConfigurationError


class TestIngestRequestValidation:
    """Tests for IngestRequest schema validation."""
    
    def test_valid_request(self):
        """Test that a valid request passes validation."""
        request = IngestRequest(
            corpus="public_domain_gov",
            async_ingest=False
        )
        assert request.corpus == "public_domain_gov"
        assert request.async_ingest is False
    
    def test_valid_request_with_defaults(self):
        """Test that defaults are applied correctly."""
        request = IngestRequest(corpus="my_corpus")
        assert request.async_ingest is False
    
    def test_invalid_request_empty_corpus(self):
        """Test that empty corpus fails validation."""
        with pytest.raises(ValueError):
            IngestRequest(corpus="")
    
    def test_invalid_request_whitespace_corpus(self):
        """Test that whitespace-only corpus fails validation."""
        with pytest.raises(ValueError):
            IngestRequest(corpus="   ")
    
    def test_invalid_request_too_long_corpus(self):
        """Test that corpus over 100 chars fails validation."""
        long_corpus = "a" * 101
        with pytest.raises(ValueError):
            IngestRequest(corpus=long_corpus)
    
    def test_valid_request_async_true(self):
        """Test that async_ingest can be set to True."""
        request = IngestRequest(corpus="test", async_ingest=True)
        assert request.async_ingest is True


class TestIngestResponseFormat:
    """Tests for IngestResponse schema format."""
    
    def test_response_with_success(self):
        """Test response with successful ingestion."""
        response = IngestResponse(
            status="completed",
            documents_count=42,
            chunks_count=1250,
            errors=[]
        )
        assert response.status == "completed"
        assert response.documents_count == 42
        assert response.chunks_count == 1250
        assert response.errors == []
    
    def test_response_with_errors(self):
        """Test response with errors."""
        response = IngestResponse(
            status="failed",
            documents_count=10,
            chunks_count=50,
            errors=["Failed to process doc-003: encoding error"]
        )
        assert response.status == "failed"
        assert len(response.errors) == 1
        assert "encoding error" in response.errors[0]
    
    def test_response_defaults(self):
        """Test response with default values."""
        response = IngestResponse()
        assert response.status == "completed"
        assert response.documents_count == 0
        assert response.chunks_count == 0
        assert response.errors == []
    
    def test_response_in_progress(self):
        """Test response with in_progress status."""
        response = IngestResponse(
            status="in_progress",
            documents_count=0,
            chunks_count=0,
            errors=[]
        )
        assert response.status == "in_progress"


class TestIngestEndpoint:
    """Tests for the POST /ingest endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock ingestion pipeline that returns a successful result."""
        mock = MagicMock()
        mock.run.return_value = IngestResult(
            documents_count=42,
            chunks_count=1250,
            errors=[]
        )
        return mock
    
    @pytest.fixture
    def mock_pipeline_with_errors(self):
        """Create a mock pipeline that returns errors."""
        mock = MagicMock()
        mock.run.return_value = IngestResult(
            documents_count=10,
            chunks_count=50,
            errors=["Failed to process doc-003: encoding error"]
        )
        return mock
    
    def test_ingest_endpoint_success(self, client, mock_pipeline):
        """Test successful ingestion."""
        with patch("src.app.routes.ingest.get_ingestion_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/ingest",
                json={
                    "corpus": "public_domain_gov",
                    "async_ingest": False
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["documents_count"] == 42
        assert data["chunks_count"] == 1250
        assert data["errors"] == []
    
    def test_ingest_endpoint_with_errors(self, client, mock_pipeline_with_errors):
        """Test ingestion with errors."""
        with patch("src.app.routes.ingest.get_ingestion_pipeline", return_value=mock_pipeline_with_errors):
            response = client.post(
                "/ingest",
                json={
                    "corpus": "test_corpus",
                    "async_ingest": False
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "failed"
        assert data["documents_count"] == 10
        assert data["chunks_count"] == 50
        assert len(data["errors"]) == 1
    
    def test_ingest_endpoint_async(self, client):
        """Test async ingestion starts successfully."""
        with patch("src.app.routes.ingest.asyncio.create_task") as mock_create_task:
            response = client.post(
                "/ingest",
                json={
                    "corpus": "test_corpus",
                    "async_ingest": True
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "in_progress"
        assert data["documents_count"] == 0
        assert data["chunks_count"] == 0
        mock_create_task.assert_called_once()
    
    def test_ingest_endpoint_empty_corpus(self, client):
        """Test ingestion with empty corpus name."""
        response = client.post(
            "/ingest",
            json={
                "corpus": "",
                "async_ingest": False
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_ingest_endpoint_missing_corpus(self, client):
        """Test ingestion with missing corpus name."""
        response = client.post(
            "/ingest",
            json={
                "async_ingest": False
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_ingest_endpoint_configuration_error(self, client):
        """Test ingestion with configuration error."""
        with patch("src.app.routes.ingest.get_ingestion_pipeline") as mock_get_pipeline:
            mock_get_pipeline.side_effect = ConfigurationError(
                "Invalid configuration",
                details="Unknown connector type"
            )
            
            response = client.post(
                "/ingest",
                json={
                    "corpus": "test_corpus",
                    "async_ingest": False
                }
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_ingest_endpoint_rag_error(self, client):
        """Test ingestion with RAG error."""
        with patch("src.app.routes.ingest.get_ingestion_pipeline") as mock_get_pipeline:
            mock_get_pipeline.side_effect = RAGError(
                "Ingestion failed",
                details="Connection refused"
            )
            
            response = client.post(
                "/ingest",
                json={
                    "corpus": "test_corpus",
                    "async_ingest": False
                }
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_ingest_endpoint_unexpected_error(self, client):
        """Test ingestion with unexpected error."""
        with patch("src.app.routes.ingest.get_ingestion_pipeline") as mock_get_pipeline:
            mock_get_pipeline.side_effect = Exception("Unexpected error")
            
            response = client.post(
                "/ingest",
                json={
                    "corpus": "test_corpus",
                    "async_ingest": False
                }
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestIngestStatusEndpoint:
    """Tests for the GET /ingest/status/{corpus} endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_status_endpoint_not_found(self, client):
        """Test status endpoint when no ingestion exists."""
        # Clear any existing status
        from src.app.routes.ingest import _ingestion_status
        _ingestion_status.clear()
        
        response = client.get("/ingest/status/nonexistent_corpus")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_status_endpoint_with_ingestion(self, client):
        """Test status endpoint with existing ingestion."""
        from src.app.routes.ingest import _ingestion_status
        
        # Set up a mock status
        _ingestion_status["test_corpus"] = IngestResponse(
            status="completed",
            documents_count=10,
            chunks_count=100,
            errors=[]
        )
        
        response = client.get("/ingest/status/test_corpus")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["documents_count"] == 10
        assert data["chunks_count"] == 100
        
        # Clean up
        _ingestion_status.clear()
    
    def test_status_endpoint_in_progress(self, client):
        """Test status endpoint with in_progress status."""
        from src.app.routes.ingest import _ingestion_status
        
        # Set up a mock in_progress status
        _ingestion_status["processing_corpus"] = IngestResponse(
            status="in_progress",
            documents_count=0,
            chunks_count=0,
            errors=[]
        )
        
        response = client.get("/ingest/status/processing_corpus")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "in_progress"
        
        # Clean up
        _ingestion_status.clear()


class TestGetIngestionPipeline:
    """Tests for the get_ingestion_pipeline function."""
    
    def test_get_pipeline_creates_pipeline(self):
        """Test that get_ingestion_pipeline creates a pipeline."""
        from src.app.routes.ingest import get_ingestion_pipeline
        
        with patch("src.app.routes.ingest.ConnectorFactory.get_connector") as mock_connector:
            mock_connector.return_value = MagicMock()
            
            pipeline = get_ingestion_pipeline("test_corpus")
            
            assert pipeline is not None
            assert hasattr(pipeline, 'run')
            mock_connector.assert_called_once()
    
    def test_get_pipeline_with_invalid_config(self):
        """Test get_ingestion_pipeline with invalid configuration."""
        from src.app.routes.ingest import get_ingestion_pipeline
        
        with patch("src.app.routes.ingest.CorpusConfig") as mock_config:
            mock_config.side_effect = ValueError("Invalid configuration")
            
            with pytest.raises(Exception):
                get_ingestion_pipeline("test_corpus")


class TestIngestEndpointIntegration:
    """Integration tests for the ingest endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_full_ingestion_flow(self, client):
        """Test full ingestion flow with mocked components."""
        mock_result = IngestResult(
            documents_count=5,
            chunks_count=25,
            errors=[]
        )
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result
        
        with patch("src.app.routes.ingest.get_ingestion_pipeline", return_value=mock_pipeline):
            # Start ingestion
            response = client.post(
                "/ingest",
                json={
                    "corpus": "integration_test",
                    "async_ingest": False
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "completed"
            assert data["documents_count"] == 5
            assert data["chunks_count"] == 25
    
    def test_ingestion_with_partial_failures(self, client):
        """Test ingestion with some documents failing."""
        mock_result = IngestResult(
            documents_count=8,
            chunks_count=40,
            errors=[
                "Failed to process doc-003: encoding error",
                "Failed to process doc-007: file not found"
            ]
        )
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result
        
        with patch("src.app.routes.ingest.get_ingestion_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/ingest",
                json={
                    "corpus": "partial_failure_test",
                    "async_ingest": False
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "failed"
            assert len(data["errors"]) == 2
