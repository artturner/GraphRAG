"""Tests for the health check endpoint.

This module tests the health check endpoint, including:
- Health endpoint returns 200 when all components are healthy
- Health check with unhealthy component returns appropriate status
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app import app


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client.
        
        Returns:
            TestClient instance for the FastAPI app.
        """
        return TestClient(app)
    
    def test_health_endpoint_returns_200(self, client: TestClient):
        """Test that health endpoint returns 200 status.
        
        Args:
            client: The test client.
        """
        with patch("src.app.routes.health.check_embeddings_health") as mock_embeddings, \
             patch("src.app.routes.health.check_vector_store_health") as mock_vector_store, \
             patch("src.app.routes.health.check_llm_health") as mock_llm:
            
            # Mock all health checks to return healthy
            mock_embeddings.return_value = {"status": "ok"}
            mock_vector_store.return_value = {"status": "ok"}
            mock_llm.return_value = {"status": "ok"}
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "healthy"
            assert data["components"]["embeddings"] == "ok"
            assert data["components"]["vector_store"] == "ok"
            assert data["components"]["llm"] == "ok"
            assert "version" in data
    
    def test_health_endpoint_with_unhealthy_embeddings(self, client: TestClient):
        """Test health endpoint when embeddings component is unhealthy.
        
        Args:
            client: The test client.
        """
        with patch("src.app.routes.health.check_embeddings_health") as mock_embeddings, \
             patch("src.app.routes.health.check_vector_store_health") as mock_vector_store, \
             patch("src.app.routes.health.check_llm_health") as mock_llm:
            
            # Mock embeddings as unhealthy
            mock_embeddings.return_value = {"status": "error", "message": "Connection failed"}
            mock_vector_store.return_value = {"status": "ok"}
            mock_llm.return_value = {"status": "ok"}
            
            response = client.get("/health")
            
            # Health endpoint still returns 200 but with unhealthy status
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["components"]["embeddings"] == "error"
            assert data["components"]["vector_store"] == "ok"
            assert data["components"]["llm"] == "ok"
    
    def test_health_endpoint_with_unhealthy_vector_store(self, client: TestClient):
        """Test health endpoint when vector store component is unhealthy.
        
        Args:
            client: The test client.
        """
        with patch("src.app.routes.health.check_embeddings_health") as mock_embeddings, \
             patch("src.app.routes.health.check_vector_store_health") as mock_vector_store, \
             patch("src.app.routes.health.check_llm_health") as mock_llm:
            
            # Mock vector store as unhealthy
            mock_embeddings.return_value = {"status": "ok"}
            mock_vector_store.return_value = {"status": "error", "message": "Database unavailable"}
            mock_llm.return_value = {"status": "ok"}
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["components"]["embeddings"] == "ok"
            assert data["components"]["vector_store"] == "error"
            assert data["components"]["llm"] == "ok"
    
    def test_health_endpoint_with_unhealthy_llm(self, client: TestClient):
        """Test health endpoint when LLM component is unhealthy.
        
        Args:
            client: The test client.
        """
        with patch("src.app.routes.health.check_embeddings_health") as mock_embeddings, \
             patch("src.app.routes.health.check_vector_store_health") as mock_vector_store, \
             patch("src.app.routes.health.check_llm_health") as mock_llm:
            
            # Mock LLM as unhealthy
            mock_embeddings.return_value = {"status": "ok"}
            mock_vector_store.return_value = {"status": "ok"}
            mock_llm.return_value = {"status": "error", "message": "API key invalid"}
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["components"]["embeddings"] == "ok"
            assert data["components"]["vector_store"] == "ok"
            assert data["components"]["llm"] == "error"
    
    def test_health_endpoint_all_components_unhealthy(self, client: TestClient):
        """Test health endpoint when all components are unhealthy.
        
        Args:
            client: The test client.
        """
        with patch("src.app.routes.health.check_embeddings_health") as mock_embeddings, \
             patch("src.app.routes.health.check_vector_store_health") as mock_vector_store, \
             patch("src.app.routes.health.check_llm_health") as mock_llm:
            
            # Mock all components as unhealthy
            mock_embeddings.return_value = {"status": "error", "message": "Failed"}
            mock_vector_store.return_value = {"status": "error", "message": "Failed"}
            mock_llm.return_value = {"status": "error", "message": "Failed"}
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["components"]["embeddings"] == "error"
            assert data["components"]["vector_store"] == "error"
            assert data["components"]["llm"] == "error"
    
    def test_health_endpoint_has_version(self, client: TestClient):
        """Test that health endpoint includes version in response.
        
        Args:
            client: The test client.
        """
        with patch("src.app.routes.health.check_embeddings_health") as mock_embeddings, \
             patch("src.app.routes.health.check_vector_store_health") as mock_vector_store, \
             patch("src.app.routes.health.check_llm_health") as mock_llm:
            
            mock_embeddings.return_value = {"status": "ok"}
            mock_vector_store.return_value = {"status": "ok"}
            mock_llm.return_value = {"status": "ok"}
            
            response = client.get("/health")
            
            data = response.json()
            assert "version" in data
            assert data["version"] == "1.0.0"


class TestHealthCheckFunctions:
    """Tests for individual health check functions."""
    
    def test_check_embeddings_health_success(self):
        """Test embeddings health check when successful."""
        import asyncio
        from src.app.routes.health import check_embeddings_health
        
        with patch("src.embeddings.factory.EmbeddingsFactory") as mock_factory:
            mock_embeddings = MagicMock()
            mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_factory.get_embeddings.return_value = mock_embeddings
            
            result = asyncio.run(check_embeddings_health())
            
            assert result["status"] == "ok"
    
    def test_check_embeddings_health_failure(self):
        """Test embeddings health check when it fails."""
        import asyncio
        from src.app.routes.health import check_embeddings_health
        
        with patch("src.embeddings.factory.EmbeddingsFactory") as mock_factory:
            mock_factory.get_embeddings.side_effect = Exception("Connection failed")
            
            result = asyncio.run(check_embeddings_health())
            
            assert result["status"] == "error"
            assert "Connection failed" in result["message"]
    
    def test_check_vector_store_health_success(self):
        """Test vector store health check when successful."""
        import asyncio
        from src.app.routes.health import check_vector_store_health
        
        with patch("src.store.factory.VectorStoreFactory") as mock_factory:
            mock_store = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 10
            mock_store._collection = mock_collection
            mock_factory.get_store.return_value = mock_store
            
            result = asyncio.run(check_vector_store_health())
            
            assert result["status"] == "ok"
    
    def test_check_vector_store_health_failure(self):
        """Test vector store health check when it fails."""
        import asyncio
        from src.app.routes.health import check_vector_store_health
        
        with patch("src.store.factory.VectorStoreFactory") as mock_factory:
            mock_factory.get_store.side_effect = Exception("Database unavailable")
            
            result = asyncio.run(check_vector_store_health())
            
            assert result["status"] == "error"
            assert "Database unavailable" in result["message"]
    
    def test_check_llm_health_success(self):
        """Test LLM health check when successful."""
        import asyncio
        from src.app.routes.health import check_llm_health
        
        with patch("src.llm.factory.LLMFactory") as mock_factory:
            mock_llm = MagicMock()
            mock_llm.model_name = "gpt-4"
            mock_factory.get_llm.return_value = mock_llm
            
            result = asyncio.run(check_llm_health())
            
            assert result["status"] == "ok"
    
    def test_check_llm_health_failure(self):
        """Test LLM health check when it fails."""
        import asyncio
        from src.app.routes.health import check_llm_health
        
        with patch("src.llm.factory.LLMFactory") as mock_factory:
            mock_factory.get_llm.side_effect = Exception("API key invalid")
            
            result = asyncio.run(check_llm_health())
            
            assert result["status"] == "error"
            assert "API key invalid" in result["message"]
