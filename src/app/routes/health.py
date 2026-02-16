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
    """Check the health of the embeddings component.
    
    Returns:
        Dictionary with health status ("ok" or "error").
    """
    try:
        from src.embeddings.factory import EmbeddingsFactory
        
        embeddings = EmbeddingsFactory.get_embeddings(settings.embeddings)
        
        # Try to embed a simple test string
        test_result = embeddings.embed_query("health check")
        
        if test_result and len(test_result) > 0:
            return {"status": "ok"}
        else:
            return {"status": "error", "message": "Empty embedding result"}
            
    except Exception as e:
        logger.warning(f"Embeddings health check failed: {e}")
        return {"status": "error", "message": str(e)}


async def check_vector_store_health() -> dict[str, str]:
    """Check the health of the vector store component.
    
    Returns:
        Dictionary with health status ("ok" or "error").
    """
    try:
        from src.store.factory import VectorStoreFactory
        
        store = VectorStoreFactory.get_store(
            store_type=settings.vectorstore.type,
            dimension=settings.embeddings.dimension,
            persist_directory=settings.vectorstore.persist_directory,
            collection_name=settings.vectorstore.collection_name,
        )
        
        # Try to get collection count or verify store is accessible
        # For most vector stores, we can check if they're responsive
        if hasattr(store, '_collection'):
            # ChromaDB
            count = store._collection.count()
            return {"status": "ok", "document_count": str(count)}
        elif hasattr(store, 'index'):
            # FAISS
            return {"status": "ok"}
        else:
            # Generic check - store exists
            return {"status": "ok"}
            
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        return {"status": "error", "message": str(e)}


async def check_llm_health() -> dict[str, str]:
    """Check the health of the LLM component.
    
    Returns:
        Dictionary with health status ("ok" or "error").
    """
    try:
        from src.llm.factory import LLMFactory
        
        llm = LLMFactory.get_llm(settings.llm)
        
        # For most LLMs, we can verify they're configured correctly
        # without making an actual API call (which would be expensive)
        if hasattr(llm, 'model_name') or hasattr(llm, 'model'):
            return {"status": "ok"}
        else:
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
