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

# ---------------------------------------------------------------------------
# Component references — populated during startup, cleaned up on shutdown
# ---------------------------------------------------------------------------

_components: dict = {}


def _init_components() -> dict:
    """Eagerly initialise core components so the first request is fast.

    Returns a dict of component instances that should be torn down later.
    """
    from src.embeddings.factory import EmbeddingsFactory
    from src.llm.factory import LLMFactory
    from src.store.factory import VectorStoreFactory
    from src.retrieval.service import RetrievalService
    from src.graphs.qna_graph import create_qna_graph

    components: dict = {}

    try:
        # Embeddings
        embeddings = EmbeddingsFactory.get_embeddings(settings.embeddings)
        components["embeddings"] = embeddings
        logger.info(
            "Embeddings initialised",
            extra={"provider": settings.embeddings.provider},
        )
    except Exception as exc:
        logger.warning("Embeddings init failed (will retry on first request): %s", exc)

    try:
        # Vector store
        store = VectorStoreFactory.get_store(
            settings.vectorstore,
            dimension=settings.embeddings.dimension,
        )
        components["store"] = store
        logger.info(
            "Vector store initialised",
            extra={"type": settings.vectorstore.type},
        )
    except Exception as exc:
        logger.warning("Vector store init failed: %s", exc)

    try:
        # LLM
        llm = LLMFactory.get_llm(settings.llm)
        components["llm"] = llm
        logger.info(
            "LLM initialised",
            extra={"provider": settings.llm.provider, "model": settings.llm.model_name},
        )
    except Exception as exc:
        logger.warning("LLM init failed (will retry on first request): %s", exc)

    # Retrieval service (requires embeddings + store)
    if "embeddings" in components and "store" in components:
        retrieval = RetrievalService(
            embeddings=components["embeddings"],
            store=components["store"],
        )
        components["retrieval"] = retrieval

        # Graph workflow (requires retrieval + llm)
        if "llm" in components:
            graph = create_qna_graph(
                retrieval=retrieval,
                llm=components["llm"],
                config=settings.graph,
            )
            components["graph"] = graph

            # Inject the pre-built workflow into the query route so it
            # doesn't have to rebuild on first request.
            import src.app.routes.query as _qmod

            _qmod._workflow = graph
            _qmod._retrieval_service = retrieval
            logger.info("Query workflow pre-initialised")

    return components


def _shutdown_components(components: dict) -> None:
    """Release resources held by components."""
    store = components.get("store")
    if store is not None and hasattr(store, "persist"):
        try:
            store.persist()
            logger.info("Vector store persisted on shutdown")
        except Exception as exc:
            logger.warning("Vector store persist failed: %s", exc)

    # Clear the global workflow reference so a fresh restart works cleanly.
    try:
        import src.app.routes.query as _qmod

        _qmod._workflow = None
        _qmod._retrieval_service = None
    except Exception:
        pass

    components.clear()
    logger.info("Components cleaned up")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application initialization and cleanup.

    Args:
        app: The FastAPI application instance.

    Yields:
        None during the application's lifetime.
    """
    global _components

    # Startup
    logger.info(
        "Starting Grounded GraphRAG Tutor API",
        extra={
            "version": "0.1.0",
            "debug": settings.debug,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model_name,
            "embeddings_provider": settings.embeddings.provider,
            "vectorstore_type": settings.vectorstore.type,
        },
    )

    _components = _init_components()

    yield

    # Shutdown
    logger.info("Shutting down Grounded GraphRAG Tutor API")
    _shutdown_components(_components)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


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
        """Handle AppHTTPException with standardized error response."""
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
        """Handle all RAG-related exceptions."""
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
        """Handle all unhandled exceptions."""
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
