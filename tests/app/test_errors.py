"""Tests for error handling, middleware, and error response format.

This module tests:
- HTTPException subclasses
- Error response models
- Error handling middleware
- Request logging middleware
- Correlation ID middleware
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.app.exceptions import (
    AppHTTPException,
    BadRequestError,
    ConfigurationError,
    ConnectorError,
    EmbeddingError,
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    ForbiddenError,
    IngestionError,
    InternalServerError,
    LLMError,
    NotFoundError,
    RetrievalError,
    UnauthorizedError,
    ValidationError,
    VectorStoreError,
)
from src.app.middleware import (
    CORRELATION_ID_HEADER,
    CorrelationIDMiddleware,
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    get_correlation_id,
)
from src.exceptions import RAGError


# Test ErrorDetail model
class TestErrorDetail:
    """Tests for the ErrorDetail model."""
    
    def test_error_detail_creation(self) -> None:
        """Test creating an ErrorDetail instance."""
        detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
            details={"key": "value"},
        )
        
        assert detail.code == "TEST_ERROR"
        assert detail.message == "Test error message"
        assert detail.details == {"key": "value"}
    
    def test_error_detail_without_details(self) -> None:
        """Test creating an ErrorDetail without details."""
        detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
        )
        
        assert detail.code == "TEST_ERROR"
        assert detail.message == "Test error message"
        assert detail.details is None
    
    def test_error_detail_model_dump(self) -> None:
        """Test serializing ErrorDetail to dict."""
        detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
            details={"key": "value"},
        )
        
        result = detail.model_dump()
        
        assert result == {
            "code": "TEST_ERROR",
            "message": "Test error message",
            "details": {"key": "value"},
        }
    
    def test_error_detail_model_dump_exclude_none(self) -> None:
        """Test serializing ErrorDetail excluding None values."""
        detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
        )
        
        result = detail.model_dump(exclude_none=True)
        
        assert result == {
            "code": "TEST_ERROR",
            "message": "Test error message",
        }
        assert "details" not in result


# Test ErrorResponse model
class TestErrorResponse:
    """Tests for the ErrorResponse model."""
    
    def test_error_response_creation(self) -> None:
        """Test creating an ErrorResponse instance."""
        error_detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
        )
        
        response = ErrorResponse(
            error=error_detail,
            correlation_id="test-correlation-id",
        )
        
        assert response.error == error_detail
        assert response.correlation_id == "test-correlation-id"
    
    def test_error_response_without_correlation_id(self) -> None:
        """Test creating an ErrorResponse without correlation ID."""
        error_detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
        )
        
        response = ErrorResponse(
            error=error_detail,
        )
        
        assert response.error == error_detail
        assert response.correlation_id is None
    
    def test_error_response_model_dump(self) -> None:
        """Test serializing ErrorResponse to dict."""
        error_detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
            details={"key": "value"},
        )
        
        response = ErrorResponse(
            error=error_detail,
            correlation_id="test-correlation-id",
        )
        
        result = response.model_dump(exclude_none=True)
        
        assert result == {
            "error": {
                "code": "TEST_ERROR",
                "message": "Test error message",
                "details": {"key": "value"},
            },
            "correlation_id": "test-correlation-id",
        }


# Test AppHTTPException
class TestAppHTTPException:
    """Tests for the AppHTTPException class."""
    
    def test_app_http_exception_creation(self) -> None:
        """Test creating an AppHTTPException."""
        exc = AppHTTPException(
            status_code=400,
            code="BAD_REQUEST",
            message="Invalid request",
            details={"field": "value"},
        )
        
        assert exc.status_code == 400
        assert exc.code == "BAD_REQUEST"
        assert exc.message == "Invalid request"
        assert exc.details == {"field": "value"}
    
    def test_app_http_exception_to_error_detail(self) -> None:
        """Test converting AppHTTPException to ErrorDetail."""
        exc = AppHTTPException(
            status_code=400,
            code="BAD_REQUEST",
            message="Invalid request",
            details={"field": "value"},
        )
        
        detail = exc.to_error_detail()
        
        assert isinstance(detail, ErrorDetail)
        assert detail.code == "BAD_REQUEST"
        assert detail.message == "Invalid request"
        assert detail.details == {"field": "value"}
    
    def test_app_http_exception_with_headers(self) -> None:
        """Test creating an AppHTTPException with custom headers."""
        exc = AppHTTPException(
            status_code=401,
            code="UNAUTHORIZED",
            message="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        assert exc.headers == {"WWW-Authenticate": "Bearer"}


# Test specific HTTP exception subclasses
class TestHTTPExceptionSubclasses:
    """Tests for specific HTTP exception subclasses."""
    
    def test_bad_request_error(self) -> None:
        """Test BadRequestError."""
        exc = BadRequestError(
            message="Invalid input",
            details={"field": "required"},
        )
        
        assert exc.status_code == 400
        assert exc.code == ErrorCodes.BAD_REQUEST
        assert exc.message == "Invalid input"
    
    def test_validation_error(self) -> None:
        """Test ValidationError."""
        exc = ValidationError(
            message="Validation failed",
            details={"field": "invalid format"},
        )
        
        assert exc.status_code == 422
        assert exc.code == ErrorCodes.VALIDATION_ERROR
        assert exc.message == "Validation failed"
    
    def test_not_found_error(self) -> None:
        """Test NotFoundError."""
        exc = NotFoundError(
            message="Document not found",
            details={"id": "123"},
        )
        
        assert exc.status_code == 404
        assert exc.code == ErrorCodes.NOT_FOUND
        assert exc.message == "Document not found"
    
    def test_unauthorized_error(self) -> None:
        """Test UnauthorizedError."""
        exc = UnauthorizedError(message="Invalid credentials")
        
        assert exc.status_code == 401
        assert exc.code == ErrorCodes.UNAUTHORIZED
        assert exc.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_forbidden_error(self) -> None:
        """Test ForbiddenError."""
        exc = ForbiddenError(message="Access denied")
        
        assert exc.status_code == 403
        assert exc.code == ErrorCodes.FORBIDDEN
    
    def test_internal_server_error(self) -> None:
        """Test InternalServerError."""
        exc = InternalServerError(
            message="Something went wrong",
            details={"trace": "abc123"},
        )
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.INTERNAL_ERROR
    
    def test_retrieval_error(self) -> None:
        """Test RetrievalError."""
        exc = RetrievalError(
            message="Failed to retrieve documents",
            details={"query": "test"},
        )
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.RETRIEVAL_ERROR
    
    def test_ingestion_error(self) -> None:
        """Test IngestionError."""
        exc = IngestionError(message="Failed to ingest")
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.INGESTION_ERROR
    
    def test_embedding_error(self) -> None:
        """Test EmbeddingError."""
        exc = EmbeddingError(message="Failed to generate embeddings")
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.EMBEDDING_ERROR
    
    def test_vector_store_error(self) -> None:
        """Test VectorStoreError."""
        exc = VectorStoreError(message="Vector store failed")
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.VECTOR_STORE_ERROR
    
    def test_llm_error(self) -> None:
        """Test LLMError."""
        exc = LLMError(message="LLM failed")
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.LLM_ERROR
    
    def test_connector_error(self) -> None:
        """Test ConnectorError."""
        exc = ConnectorError(message="Connector failed")
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.CONNECTOR_ERROR
    
    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        exc = ConfigurationError(message="Invalid config")
        
        assert exc.status_code == 500
        assert exc.code == ErrorCodes.CONFIGURATION_ERROR


# Test CorrelationIDMiddleware
class TestCorrelationIDMiddleware:
    """Tests for the CorrelationIDMiddleware."""
    
    def test_correlation_id_generated(self) -> None:
        """Test that correlation ID is generated when not provided."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"correlation_id": get_correlation_id(request)}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "correlation_id" in response.json()
        assert response.headers.get(CORRELATION_ID_HEADER) is not None
        
        # Verify it's a valid UUID
        correlation_id = response.json()["correlation_id"]
        uuid.UUID(correlation_id)  # Will raise if invalid
    
    def test_correlation_id_propagated(self) -> None:
        """Test that provided correlation ID is propagated."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"correlation_id": get_correlation_id(request)}
        
        client = TestClient(app)
        test_correlation_id = "test-correlation-123"
        response = client.get(
            "/test",
            headers={CORRELATION_ID_HEADER: test_correlation_id},
        )
        
        assert response.status_code == 200
        assert response.json()["correlation_id"] == test_correlation_id
        assert response.headers[CORRELATION_ID_HEADER] == test_correlation_id
    
    def test_correlation_id_in_response_headers(self) -> None:
        """Test that correlation ID is added to response headers."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert CORRELATION_ID_HEADER in response.headers
        assert response.headers[CORRELATION_ID_HEADER] is not None


# Test ErrorHandlingMiddleware
class TestErrorHandlingMiddleware:
    """Tests for the ErrorHandlingMiddleware."""
    
    def test_app_http_exception_handling(self) -> None:
        """Test handling of AppHTTPException via exception handler."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        # Register exception handlers
        from src.app.main import register_exception_handlers
        register_exception_handlers(app)
        
        @app.get("/error")
        async def error_endpoint():
            raise BadRequestError(
                message="Test error",
                details={"field": "value"},
            )
        
        client = TestClient(app)
        response = client.get("/error")
        
        assert response.status_code == 400
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == ErrorCodes.BAD_REQUEST
        assert data["error"]["message"] == "Test error"
        assert data["error"]["details"] == {"field": "value"}
        assert "correlation_id" in data
    
    def test_rag_error_handling(self) -> None:
        """Test handling of RAGError."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        # Register exception handlers
        from src.app.main import register_exception_handlers
        register_exception_handlers(app)
        
        @app.get("/error")
        async def error_endpoint():
            raise RAGError(
                message="RAG operation failed",
                details="Connection timeout",
            )
        
        client = TestClient(app)
        response = client.get("/error")
        
        assert response.status_code == 500
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == ErrorCodes.INTERNAL_ERROR
        assert data["error"]["message"] == "RAG operation failed"
    
    def test_unexpected_exception_handling(self) -> None:
        """Test handling of unexpected exceptions."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        # Register exception handlers
        from src.app.main import register_exception_handlers
        register_exception_handlers(app)
        
        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Unexpected error")
        
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")
        
        assert response.status_code == 500
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == ErrorCodes.INTERNAL_ERROR
        assert data["error"]["message"] == "An unexpected error occurred"
    
    def test_error_response_format(self) -> None:
        """Test that error response matches expected format."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        # Register exception handlers
        from src.app.main import register_exception_handlers
        register_exception_handlers(app)
        
        @app.get("/error")
        async def error_endpoint():
            raise RetrievalError(
                message="Failed to retrieve documents",
                details={"query": "test query", "index": "main"},
            )
        
        client = TestClient(app)
        response = client.get("/error")
        
        assert response.status_code == 500
        data = response.json()
        
        # Verify response structure
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "correlation_id" in data
        
        # Verify values
        assert data["error"]["code"] == "RETRIEVAL_ERROR"
        assert data["error"]["message"] == "Failed to retrieve documents"


# Test RequestLoggingMiddleware
class TestRequestLoggingMiddleware:
    """Tests for the RequestLoggingMiddleware."""
    
    def test_request_logging_excludes_health_paths(self) -> None:
        """Test that health paths are excluded from logging."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        client = TestClient(app)
        
        with patch("src.app.middleware.logger") as mock_logger:
            response = client.get("/health")
            
            assert response.status_code == 200
            # Info should not be called for excluded paths
            assert mock_logger.info.call_count == 0
    
    def test_request_logging_logs_requests(self) -> None:
        """Test that requests are logged."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        
        with patch("src.app.middleware.logger") as mock_logger:
            response = client.get("/api/test")
            
            assert response.status_code == 200
            # Should log request start and completion
            assert mock_logger.info.call_count >= 2
    
    def test_request_logging_includes_timing(self) -> None:
        """Test that request timing is logged."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        
        with patch("src.app.middleware.logger") as mock_logger:
            response = client.get("/api/test")
            
            assert response.status_code == 200
            
            # Check that duration_ms was logged in the completion log
            calls = mock_logger.info.call_args_list
            # Find the completion log (second call)
            completion_call = None
            for call in calls:
                extra = call[1].get("extra", {})
                if "duration_ms" in extra:
                    completion_call = call
                    break
            
            assert completion_call is not None
            assert "duration_ms" in completion_call[1]["extra"]


# Test correlation ID propagation
class TestCorrelationIDPropagation:
    """Tests for correlation ID propagation through the request lifecycle."""
    
    def test_correlation_id_in_error_response(self) -> None:
        """Test that correlation ID is included in error responses."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        # Register exception handlers
        from src.app.main import register_exception_handlers
        register_exception_handlers(app)
        
        @app.get("/error")
        async def error_endpoint():
            raise BadRequestError(message="Test error")
        
        client = TestClient(app)
        test_correlation_id = "test-correlation-456"
        response = client.get(
            "/error",
            headers={CORRELATION_ID_HEADER: test_correlation_id},
        )
        
        assert response.status_code == 400
        data = response.json()
        
        assert data["correlation_id"] == test_correlation_id
        assert response.headers[CORRELATION_ID_HEADER] == test_correlation_id
    
    def test_correlation_id_available_in_request_state(self) -> None:
        """Test that correlation ID is available in request state."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        captured_correlation_id = None
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            nonlocal captured_correlation_id
            captured_correlation_id = get_correlation_id(request)
            return {"status": "ok"}
        
        client = TestClient(app)
        test_correlation_id = "test-correlation-789"
        client.get(
            "/test",
            headers={CORRELATION_ID_HEADER: test_correlation_id},
        )
        
        assert captured_correlation_id == test_correlation_id
    
    def test_different_requests_have_different_correlation_ids(self) -> None:
        """Test that different requests get different correlation IDs."""
        app = FastAPI()
        app.add_middleware(CorrelationIDMiddleware)
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"correlation_id": get_correlation_id(request)}
        
        client = TestClient(app)
        
        response1 = client.get("/test")
        response2 = client.get("/test")
        
        id1 = response1.json()["correlation_id"]
        id2 = response2.json()["correlation_id"]
        
        assert id1 != id2


# Test ErrorCodes constants
class TestErrorCodes:
    """Tests for ErrorCodes constants."""
    
    def test_client_error_codes(self) -> None:
        """Test client error codes are defined."""
        assert ErrorCodes.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCodes.NOT_FOUND == "NOT_FOUND"
        assert ErrorCodes.UNAUTHORIZED == "UNAUTHORIZED"
        assert ErrorCodes.FORBIDDEN == "FORBIDDEN"
        assert ErrorCodes.BAD_REQUEST == "BAD_REQUEST"
    
    def test_server_error_codes(self) -> None:
        """Test server error codes are defined."""
        assert ErrorCodes.INTERNAL_ERROR == "INTERNAL_ERROR"
        assert ErrorCodes.RETRIEVAL_ERROR == "RETRIEVAL_ERROR"
        assert ErrorCodes.INGESTION_ERROR == "INGESTION_ERROR"
        assert ErrorCodes.EMBEDDING_ERROR == "EMBEDDING_ERROR"
        assert ErrorCodes.VECTOR_STORE_ERROR == "VECTOR_STORE_ERROR"
        assert ErrorCodes.LLM_ERROR == "LLM_ERROR"
        assert ErrorCodes.CONNECTOR_ERROR == "CONNECTOR_ERROR"
        assert ErrorCodes.CONFIGURATION_ERROR == "CONFIGURATION_ERROR"


# Integration tests with the main app
class TestErrorHandlingIntegration:
    """Integration tests for error handling with the main app."""
    
    def test_main_app_has_correlation_id_middleware(self) -> None:
        """Test that main app has correlation ID middleware."""
        from src.app.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert CORRELATION_ID_HEADER in response.headers
    
    def test_main_app_error_response_format(self) -> None:
        """Test that main app returns errors in correct format."""
        from src.app.main import app
        
        client = TestClient(app)
        
        # Test validation error (invalid request)
        response = client.post(
            "/query",
            json={"question": ""},  # Empty question should fail validation
        )
        
        # Should get a 422 validation error
        assert response.status_code == 422
        data = response.json()
        
        # Check error format
        assert "error" in data or "detail" in data
    
    def test_main_app_propagates_correlation_id(self) -> None:
        """Test that main app propagates correlation ID."""
        from src.app.main import app
        
        client = TestClient(app)
        test_correlation_id = "integration-test-123"
        
        response = client.get(
            "/health",
            headers={CORRELATION_ID_HEADER: test_correlation_id},
        )
        
        assert response.headers[CORRELATION_ID_HEADER] == test_correlation_id
