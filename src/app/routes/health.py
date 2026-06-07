"""Health check endpoint for the GraphRAG API.

This module provides the GET /health endpoint that checks the health of
all system components (embeddings, vector store, LLM).
"""

from typing import Any

from fastapi import APIRouter

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

# Application version
VERSION = "1.0.0"


async def check_embeddings_health() -> dict[str, str]:
    try:
        import src.app.routes.query as _qmod
        if _qmod._retrieval_service is not None:
            return {"status": "ok"}
        # Fallback before first request: verify config is valid
        from src.embeddings.factory import EmbeddingsFactory
        EmbeddingsFactory.get_embeddings(settings.embeddings)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"Embeddings health check failed: {e}")
        return {"status": "error", "message": str(e)}


async def check_vector_store_health() -> dict[str, str]:
    try:
        import src.app.routes.query as _qmod
        if _qmod._workflow is not None:
            # Workflow initialized — store loaded successfully at startup
            return {"status": "ok"}
        # Fallback: check persist directory exists and has content
        from pathlib import Path
        persist_dir = Path(settings.vectorstore.persist_directory)
        if persist_dir.exists() and any(persist_dir.iterdir()):
            return {"status": "ok"}
        return {"status": "error", "message": "Vector store directory empty or missing"}
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        return {"status": "error", "message": str(e)}


async def check_llm_health() -> dict[str, str]:
    try:
        import src.app.routes.query as _qmod
        if _qmod._workflow is not None:
            return {"status": "ok"}
        # Fallback: verify config is valid without instantiating
        from src.llm.factory import LLMFactory
        LLMFactory.get_llm(settings.llm)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        return {"status": "error", "message": str(e)}


@router.get(
    "/health",
    summary="Health check endpoint",
    description="Check the health status of all system components.",
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "components": {
                            "embeddings": "ok",
                            "vector_store": "ok",
                            "llm": "ok"
                        },
                        "version": "1.0.0"
                    }
                }
            },
        },
        503: {
            "description": "System is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "components": {
                            "embeddings": "ok",
                            "vector_store": "error",
                            "llm": "ok"
                        },
                        "version": "1.0.0"
                    }
                }
            },
        },
    },
)
async def health_check() -> dict[str, Any]:
    """Health check endpoint.
    
    Checks the health of all system components:
    - Embeddings service
    - Vector store
    - LLM provider
    
    Returns:
        Dictionary with overall status and component health details.
        Status is "healthy" if all components are "ok", otherwise "unhealthy".
    """
    # Run all health checks
    embeddings_status = await check_embeddings_health()
    vector_store_status = await check_vector_store_health()
    llm_status = await check_llm_health()
    
    # Determine overall status
    components_healthy = (
        embeddings_status.get("status") == "ok"
        and vector_store_status.get("status") == "ok"
        and llm_status.get("status") == "ok"
    )
    
    overall_status = "healthy" if components_healthy else "unhealthy"
    
    # Build response
    response = {
        "status": overall_status,
        "components": {
            "embeddings": embeddings_status.get("status", "error"),
            "vector_store": vector_store_status.get("status", "error"),
            "llm": llm_status.get("status", "error"),
        },
        "version": VERSION,
    }
    
    # Log health check result
    if overall_status == "healthy":
        logger.info("Health check passed")
    else:
        logger.warning(
            "Health check failed",
            extra={
                "embeddings": embeddings_status,
                "vector_store": vector_store_status,
                "llm": llm_status,
            }
        )
    
    return response
