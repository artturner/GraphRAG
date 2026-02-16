"""Middleware components for the FastAPI application.

This module provides middleware for:
- Error handling with standardized error responses
- Request logging with timing information
- Correlation ID propagation for request tracing
"""

import time
import uuid
from typing import Any, Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.app.exceptions import (
    AppHTTPException,
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    InternalServerError,
)
from src.config import settings
from src.exceptions import RAGError
from src.logging_config import get_logger

logger = get_logger(__name__)

# Header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware for handling correlation IDs.
    
    This middleware:
    - Extracts correlation ID from incoming request headers
    - Generates a new correlation ID if not present
    - Adds correlation ID to response headers
    - Stores correlation ID in request state for access by handlers
    
    Attributes:
        header_name: The header name for correlation ID.
    """
    
    def __init__(self, app: Any, header_name: str = CORRELATION_ID_HEADER) -> None:
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            header_name: The header name for correlation ID.
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add correlation ID.
        
        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.
            
        Returns:
            The response with correlation ID header.
        """
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Store in request state for access by handlers
        request.state.correlation_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses.
    
    This middleware logs:
    - Request method, path, and query parameters
    - Response status code and processing time
    - Client IP address
    
    Attributes:
        exclude_paths: Paths to exclude from logging (e.g., health checks).
    """
    
    def __init__(
        self,
        app: Any,
        exclude_paths: set[str] | None = None,
    ) -> None:
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            exclude_paths: Paths to exclude from logging.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or {"/health", "/health/ready", "/health/live"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log details.
        
        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.
            
        Returns:
            The response.
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Get correlation ID if available
        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_ip": client_ip,
                "correlation_id": correlation_id,
            }
        )
        
        # Time the request
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- {response.status_code} in {duration_ms:.2f}ms",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                    "correlation_id": correlation_id,
                }
            )
            
            return response
            
        except Exception as exc:
            # Calculate duration even for errors
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- {type(exc).__name__}: {str(exc)} in {duration_ms:.2f}ms",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                    "correlation_id": correlation_id,
                },
                exc_info=True,
            )
            
            # Re-raise to let error handling middleware handle it
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get the client IP address from the request.
        
        Checks X-Forwarded-For header first, then falls back to client host.
        
        Args:
            request: The incoming request.
            
        Returns:
            The client IP address.
        """
        # Check X-Forwarded-For header (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to client host
        if request.client:
            return request.client.host
        
        return "unknown"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors and returning standardized responses.
    
    This middleware catches:
    - AppHTTPException: Returns the error with appropriate status code
    - HTTPException: Converts to standardized error response
    - RAGError: Converts to appropriate HTTP error
    - Exception: Returns a generic internal server error
    
    All error responses include:
    - Error code
    - Error message
    - Optional details
    - Correlation ID for tracing
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and handle any errors.
        
        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.
            
        Returns:
            The response, or an error response if an exception occurred.
        """
        try:
            return await call_next(request)
        
        except AppHTTPException as exc:
            # Handle our custom HTTP exceptions
            return self._create_error_response(request, exc)
        
        except HTTPException as exc:
            # Handle standard FastAPI HTTPException
            return self._create_http_exception_response(request, exc)
        
        except RAGError as exc:
            # Handle RAG errors from the core library
            http_exc = self._convert_rag_error(exc)
            return self._create_error_response(request, http_exc)
        
        except Exception as exc:
            # Handle unexpected errors
            logger.exception(
                f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "correlation_id": getattr(request.state, "correlation_id", None),
                }
            )
            http_exc = InternalServerError(
                message="An unexpected error occurred",
                details={"error": str(exc)} if settings.debug else None,
            )
            return self._create_error_response(request, http_exc)
    
    def _create_error_response(
        self,
        request: Request,
        exc: AppHTTPException,
    ) -> JSONResponse:
        """Create a standardized error response.
        
        Args:
            request: The incoming request.
            exc: The HTTP exception.
            
        Returns:
            JSONResponse with error details.
        """
        correlation_id = getattr(request.state, "correlation_id", None)
        
        error_response = ErrorResponse(
            error=exc.to_error_detail(),
            correlation_id=correlation_id,
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(exclude_none=True),
            headers=exc.headers,
        )
    
    def _create_http_exception_response(
        self,
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        """Create a standardized error response for HTTPException.
        
        Args:
            request: The incoming request.
            exc: The HTTPException.
            
        Returns:
            JSONResponse with error details.
        """
        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Map status codes to error codes
        code_mapping = {
            400: ErrorCodes.BAD_REQUEST,
            401: ErrorCodes.UNAUTHORIZED,
            403: ErrorCodes.FORBIDDEN,
            404: ErrorCodes.NOT_FOUND,
            422: ErrorCodes.VALIDATION_ERROR,
        }
        
        code = code_mapping.get(exc.status_code, ErrorCodes.INTERNAL_ERROR)
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=str(exc.detail) if exc.detail else "An error occurred",
            ),
            correlation_id=correlation_id,
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(exclude_none=True),
            headers=exc.headers,
        )
    
    def _convert_rag_error(self, exc: RAGError) -> AppHTTPException:
        """Convert a RAGError to an appropriate HTTP exception.
        
        Args:
            exc: The RAG error.
            
        Returns:
            The corresponding HTTP exception.
        """
        from src.app.exceptions import (
            ConnectorError,
            EmbeddingError,
            IngestionError,
            LLMError,
            RetrievalError,
            VectorStoreError,
            ConfigurationError as HTTPConfigurationError,
        )
        from src.exceptions import (
            ConnectorError as RAGConnectorError,
            EmbeddingError as RAGEmbeddingError,
            IngestionError as RAGIngestionError,
            LLMError as RAGLLMError,
            RetrievalError as RAGRetrievalError,
            VectorStoreError as RAGVectorStoreError,
            ConfigurationError as RAGConfigurationError,
        )
        
        # Map RAG errors to HTTP errors
        error_mapping: dict[type[RAGError], type[AppHTTPException]] = {
            RAGConnectorError: ConnectorError,
            RAGEmbeddingError: EmbeddingError,
            RAGIngestionError: IngestionError,
            RAGLLMError: LLMError,
            RAGRetrievalError: RetrievalError,
            RAGVectorStoreError: VectorStoreError,
            RAGConfigurationError: HTTPConfigurationError,
        }
        
        http_exc_class = error_mapping.get(type(exc), InternalServerError)
        
        details = {"original_details": exc.details} if exc.details else None
        
        return http_exc_class(
            message=exc.message,
            details=details,
        )


def get_correlation_id(request: Request) -> str | None:
    """Get the correlation ID from the request state.
    
    Args:
        request: The incoming request.
        
    Returns:
        The correlation ID, or None if not set.
    """
    return getattr(request.state, "correlation_id", None)
