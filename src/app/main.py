"""FastAPI application main module.

This module provides the FastAPI application instance with all configuration,
middleware, exception handlers, and lifespan management.

Usage:
    from src.app import app
    
    # Run with: uvicorn src.app.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app.middleware import (
    CorrelationIDMiddleware,
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
)
from src.app.routes.health import router as health_router
from src.app.routes.ingest import router as ingest_router
from src.app.routes.query import router as query_router
from src.config import settings
from src.exceptions import RAGError
from src.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application initialization and cleanup.
    
    This handles startup and shutdown events for the FastAPI application,
    including initializing connections and resources.
    
    Args:
        app: The FastAPI application instance.
        
    Yields:
        None during the application's lifetime.
    """
    # Startup
    logger.info(
        "Starting Grounded GraphRAG Tutor API",
        extra={
            "debug": settings.debug,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model_name,
            "embeddings_provider": settings.embeddings.provider,
            "vectorstore_type": settings.vectorstore.type,
        }
    )
    
    # Initialize any resources here (e.g., vector store connections)
    # This is where you would preload models, establish connections, etc.
    
    yield
    
    # Shutdown
    logger.info("Shutting down Grounded GraphRAG Tutor API")
    # Cleanup resources here


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Grounded GraphRAG Tutor API",
        description=(
            "A Graph-based Retrieval-Augmented Generation (GraphRAG) system "
            "for educational tutoring with grounded, citation-backed responses. "
            "This API provides endpoints for querying the knowledge base, "
            "managing documents, and evaluating responses."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # OpenAPI metadata
        openapi_tags=[
            {
                "name": "query",
                "description": "Query operations for the GraphRAG system.",
            },
            {
                "name": "documents",
                "description": "Document management operations.",
            },
            {
                "name": "health",
                "description": "Health check endpoints.",
            },
        ],
    )
    
    # Add middleware (order matters - last added is first executed)
    # Error handling should be outermost to catch all exceptions
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Request logging for observability
    app.add_middleware(RequestLoggingMiddleware)
    
    # Correlation ID for request tracing
    app.add_middleware(CorrelationIDMiddleware)
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers (for any exceptions not caught by middleware)
    register_exception_handlers(app)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    
    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers for the application.
    
    These handlers serve as a fallback for exceptions that escape the
    ErrorHandlingMiddleware.
    
    Args:
        app: The FastAPI application instance.
    """
    from src.app.exceptions import (
        AppHTTPException,
        ErrorDetail,
        ErrorResponse,
        InternalServerError,
    )
    from src.app.middleware import get_correlation_id
    
    @app.exception_handler(AppHTTPException)
    async def app_http_exception_handler(
        request: Request, exc: AppHTTPException
    ) -> JSONResponse:
        """Handle AppHTTPException with standardized error response.
        
        Args:
            request: The incoming request.
            exc: The raised AppHTTPException.
            
        Returns:
            JSONResponse with error details.
        """
        correlation_id = get_correlation_id(request)
        
        logger.error(
            f"AppHTTPException: {exc.code} - {exc.message}",
            extra={
                "code": exc.code,
                "details": exc.details,
                "path": request.url.path,
                "correlation_id": correlation_id,
            }
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ),
            correlation_id=correlation_id,
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(exclude_none=True),
            headers=exc.headers,
        )
    
    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
        """Handle all RAG-related exceptions.
        
        Args:
            request: The incoming request.
            exc: The raised RAGError exception.
            
        Returns:
            JSONResponse with error details.
        """
        correlation_id = get_correlation_id(request)
        
        logger.error(
            f"RAGError: {exc.message}",
            extra={
                "details": exc.details,
                "path": request.url.path,
                "correlation_id": correlation_id,
            }
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message=exc.message,
                details={"original_details": exc.details} if exc.details else None,
            ),
            correlation_id=correlation_id,
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(exclude_none=True),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all unhandled exceptions.
        
        Args:
            request: The incoming request.
            exc: The raised exception.
            
        Returns:
            JSONResponse with error details.
        """
        correlation_id = get_correlation_id(request)
        
        logger.exception(
            f"Unhandled exception: {str(exc)}",
            extra={
                "path": request.url.path,
                "correlation_id": correlation_id,
            }
        )
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                details={"error": str(exc)} if settings.debug else None,
            ),
            correlation_id=correlation_id,
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(exclude_none=True),
        )


# Create the application instance
app = create_app()
