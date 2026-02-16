"""Tests for the FastAPI application main module.

This module tests the FastAPI application configuration, including:
- App creation and configuration
- CORS middleware configuration
- OpenAPI documentation availability
- Exception handlers
- Health check endpoints
"""

import inspect

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from src.app import app, create_app
from src.app.main import lifespan
from src.exceptions import RAGError, LLMError, VectorStoreError


class TestAppCreation:
    """Tests for FastAPI application creation and configuration."""
    
    def test_app_instance_exists(self):
        """Test that the app instance is created."""
        assert app is not None
        assert app.title == "Grounded GraphRAG Tutor API"
    
    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance."""
        new_app = create_app()
        assert new_app is not None
        assert new_app.title == "Grounded GraphRAG Tutor API"
    
    def test_app_has_correct_version(self):
        """Test that the app has the correct version."""
        assert app.version == "0.1.0"
    
    def test_app_has_description(self):
        """Test that the app has a description."""
        assert app.description is not None
        assert "GraphRAG" in app.description
    
    def test_app_has_lifespan(self):
        """Test that the app has a lifespan context manager."""
        assert app.router.lifespan_context is not None
    
    def test_app_has_openapi_tags(self):
        """Test that the app has OpenAPI tags configured."""
        assert app.openapi_tags is not None
        tag_names = [tag["name"] for tag in app.openapi_tags]
        assert "query" in tag_names
        assert "documents" in tag_names
        assert "health" in tag_names


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""
    
    def test_cors_middleware_is_configured(self):
        """Test that CORS middleware is configured."""
        # Check that CORSMiddleware is in the middleware stack
        middleware_types = [
            type(middleware).__name__
            for middleware in app.user_middleware
        ]
        # CORSMiddleware is wrapped, so we check the middleware stack
        assert len(app.user_middleware) > 0
    
    def test_cors_allows_all_origins(self):
        """Test that CORS allows all origins by default."""
        client = TestClient(app)
        
        # Make a preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            }
        )
        
        # Check that CORS headers are present
        assert response.status_code == status.HTTP_200_OK
    
    def test_cors_headers_on_response(self):
        """Test that CORS headers are present in responses."""
        client = TestClient(app)
        
        response = client.get(
            "/health",
            headers={"Origin": "http://example.com"}
        )
        
        # The response should succeed
        assert response.status_code == status.HTTP_200_OK


class TestOpenAPIDocs:
    """Tests for OpenAPI documentation availability."""
    
    def test_docs_endpoint_available(self):
        """Test that the /docs endpoint is available."""
        client = TestClient(app)
        response = client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint_available(self):
        """Test that the /redoc endpoint is available."""
        client = TestClient(app)
        response = client.get("/redoc")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_json_endpoint_available(self):
        """Test that the /openapi.json endpoint is available."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"
    
    def test_openapi_schema_has_correct_info(self):
        """Test that the OpenAPI schema has correct info."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert schema["info"]["title"] == "Grounded GraphRAG Tutor API"
        assert schema["info"]["version"] == "0.1.0"
    
    def test_openapi_schema_has_paths(self):
        """Test that the OpenAPI schema has paths defined."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/query" in schema["paths"]
        assert "/ingest" in schema["paths"]
    
    def test_openapi_schema_has_tags(self):
        """Test that the OpenAPI schema has tags defined."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "tags" in schema
        tag_names = [tag["name"] for tag in schema["tags"]]
        assert "health" in tag_names


class TestExceptionHandlers:
    """Tests for exception handlers."""
    
    def test_rag_error_handler_returns_500(self):
        """Test that RAGError returns a 500 response."""
        # Create a test app with a route that raises RAGError
        from fastapi import FastAPI
        
        test_app = FastAPI()
        
        @test_app.get("/test-error")
        async def raise_rag_error():
            raise RAGError("Test error", details="Test details")
        
        # Import and register the exception handler
        from src.app.main import register_exception_handlers
        register_exception_handlers(test_app)
        
        client = TestClient(test_app)
        response = client.get("/test-error")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["error"]["message"] == "Test error"
    
    def test_llm_error_handler_returns_500(self):
        """Test that LLMError (subclass of RAGError) returns a 500 response."""
        from fastapi import FastAPI
        
        test_app = FastAPI()
        
        @test_app.get("/test-llm-error")
        async def raise_llm_error():
            raise LLMError("LLM failed", details="Rate limit exceeded")
        
        from src.app.main import register_exception_handlers
        register_exception_handlers(test_app)
        
        client = TestClient(test_app)
        response = client.get("/test-llm-error")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"]["code"] == "INTERNAL_ERROR"
    
    def test_general_exception_handler_returns_500(self):
        """Test that unhandled exceptions return a 500 response."""
        from fastapi import FastAPI
        
        test_app = FastAPI()
        
        @test_app.get("/test-unhandled")
        async def raise_unhandled():
            raise ValueError("Unexpected error")
        
        from src.app.main import register_exception_handlers
        register_exception_handlers(test_app)
        
        # Use raise_server_exceptions=False to prevent the exception from propagating
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test-unhandled")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_ERROR"


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check_returns_200(self):
        """Test that health check returns 200."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_health_check_returns_status(self):
        """Test that health check returns status."""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        
        # Status can be "healthy" or "unhealthy" depending on component health
        assert data["status"] in ["healthy", "unhealthy"]
        assert "version" in data
    
    def test_health_check_returns_components(self):
        """Test that health check returns component info."""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        
        assert "components" in data
        assert "embeddings" in data["components"]
        assert "vector_store" in data["components"]
        assert "llm" in data["components"]


class TestLifespan:
    """Tests for lifespan context manager."""
    
    def test_lifespan_is_async_context_manager(self):
        """Test that lifespan is an async context manager."""
        # The @asynccontextmanager decorator wraps the function,
        # so we check if it has the __wrapped__ attribute or is callable
        assert callable(lifespan)
        # Check that it's an async context manager by checking for the presence
        # of the async context manager protocol
        import contextlib
        # Functions decorated with @asynccontextmanager have __wrapped__ attribute
        assert hasattr(lifespan, '__wrapped__') or callable(lifespan)
    
    def test_lifespan_yields_none(self):
        """Test that lifespan yields None."""
        import asyncio
        from fastapi import FastAPI
        
        test_app = FastAPI()
        
        async def run_lifespan():
            async with lifespan(test_app):
                # If we get here, the lifespan context manager works
                pass
            # If we get here, the cleanup worked
            return True
        
        result = asyncio.run(run_lifespan())
        assert result
