"""HTTP exceptions and error response models for the FastAPI application.

This module provides HTTPException subclasses for different error types,
along with standardized error response models for consistent API responses.
"""

from typing import Any, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel, Field


# Error code constants
class ErrorCodes:
    """Constants for error codes used in API responses."""
    
    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    BAD_REQUEST = "BAD_REQUEST"
    
    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    INGESTION_ERROR = "INGESTION_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    LLM_ERROR = "LLM_ERROR"
    CONNECTOR_ERROR = "CONNECTOR_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class ErrorDetail(BaseModel):
    """Detailed error information for API responses.
    
    Attributes:
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Optional additional details about the error.
    """
    
    code: str = Field(
        ...,
        description="Machine-readable error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional additional details about the error"
    )


class ErrorResponse(BaseModel):
    """Standard error response format for API errors.
    
    This model ensures consistent error responses across all endpoints.
    
    Attributes:
        error: The error details.
        correlation_id: Unique identifier for request tracing.
    
    Example:
        ```json
        {
            "error": {
                "code": "RETRIEVAL_ERROR",
                "message": "Failed to retrieve documents",
                "details": {"query": "test query"}
            },
            "correlation_id": "abc-123"
        }
        ```
    """
    
    error: ErrorDetail = Field(
        ...,
        description="The error details"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for request tracing"
    )


# HTTP Exception subclasses
class AppHTTPException(HTTPException):
    """Base HTTP exception for the application.
    
    This extends FastAPI's HTTPException with additional fields for
    structured error responses.
    
    Attributes:
        status_code: HTTP status code.
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Optional additional details.
        headers: Optional HTTP headers.
    """
    
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialize the HTTP exception.
        
        Args:
            status_code: HTTP status code.
            code: Machine-readable error code.
            message: Human-readable error message.
            details: Optional additional details.
            headers: Optional HTTP headers.
        """
        self.code = code
        self.message = message
        self.details = details
        super().__init__(
            status_code=status_code,
            detail=message,
            headers=headers
        )
    
    def to_error_detail(self) -> ErrorDetail:
        """Convert to ErrorDetail model.
        
        Returns:
            ErrorDetail instance.
        """
        return ErrorDetail(
            code=self.code,
            message=self.message,
            details=self.details
        )


# Client errors (4xx)
class BadRequestError(AppHTTPException):
    """Exception for bad request errors (400)."""
    
    def __init__(
        self,
        message: str = "Bad request",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the bad request error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=ErrorCodes.BAD_REQUEST,
            message=message,
            details=details
        )


class ValidationError(AppHTTPException):
    """Exception for validation errors (422)."""
    
    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the validation error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code=ErrorCodes.VALIDATION_ERROR,
            message=message,
            details=details
        )


class NotFoundError(AppHTTPException):
    """Exception for not found errors (404)."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the not found error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code=ErrorCodes.NOT_FOUND,
            message=message,
            details=details
        )


class UnauthorizedError(AppHTTPException):
    """Exception for unauthorized errors (401)."""
    
    def __init__(
        self,
        message: str = "Unauthorized",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the unauthorized error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ErrorCodes.UNAUTHORIZED,
            message=message,
            details=details,
            headers={"WWW-Authenticate": "Bearer"}
        )


class ForbiddenError(AppHTTPException):
    """Exception for forbidden errors (403)."""
    
    def __init__(
        self,
        message: str = "Forbidden",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the forbidden error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            code=ErrorCodes.FORBIDDEN,
            message=message,
            details=details
        )


# Server errors (5xx)
class InternalServerError(AppHTTPException):
    """Exception for internal server errors (500)."""
    
    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the internal server error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.INTERNAL_ERROR,
            message=message,
            details=details
        )


class RetrievalError(AppHTTPException):
    """Exception for retrieval errors (500)."""
    
    def __init__(
        self,
        message: str = "Failed to retrieve documents",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the retrieval error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.RETRIEVAL_ERROR,
            message=message,
            details=details
        )


class IngestionError(AppHTTPException):
    """Exception for ingestion errors (500)."""
    
    def __init__(
        self,
        message: str = "Failed to ingest documents",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the ingestion error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.INGESTION_ERROR,
            message=message,
            details=details
        )


class EmbeddingError(AppHTTPException):
    """Exception for embedding errors (500)."""
    
    def __init__(
        self,
        message: str = "Failed to generate embeddings",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the embedding error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.EMBEDDING_ERROR,
            message=message,
            details=details
        )


class VectorStoreError(AppHTTPException):
    """Exception for vector store errors (500)."""
    
    def __init__(
        self,
        message: str = "Vector store operation failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the vector store error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.VECTOR_STORE_ERROR,
            message=message,
            details=details
        )


class LLMError(AppHTTPException):
    """Exception for LLM errors (500)."""
    
    def __init__(
        self,
        message: str = "LLM operation failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the LLM error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.LLM_ERROR,
            message=message,
            details=details
        )


class ConnectorError(AppHTTPException):
    """Exception for connector errors (500)."""
    
    def __init__(
        self,
        message: str = "Document connector failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the connector error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.CONNECTOR_ERROR,
            message=message,
            details=details
        )


class ConfigurationError(AppHTTPException):
    """Exception for configuration errors (500)."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the configuration error.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details.
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCodes.CONFIGURATION_ERROR,
            message=message,
            details=details
        )
