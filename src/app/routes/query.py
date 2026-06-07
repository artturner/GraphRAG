"""Query endpoint for the GraphRAG API.

This module provides the POST /query endpoint that processes user questions
through the LangGraph workflow and returns structured responses.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.auth import verify_api_key
from src.app.schemas import QueryRequest, QueryResponse
from src.config import settings
from src.exceptions import RAGError
from src.graphs.qna_graph import create_qna_graph
from src.graphs.state import GraphState
from src.logging_config import get_logger
from src.retrieval.service import RetrievalService

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


# Global workflow instance (lazy initialization)
_workflow: Any = None
_retrieval_service: RetrievalService | None = None


def get_workflow() -> Any:
    """Get or create the LangGraph workflow instance.
    
    Returns:
        The compiled LangGraph workflow.
    """
    global _workflow
    
    if _workflow is None:
        # Import here to avoid circular imports
        from src.embeddings.factory import EmbeddingsFactory
        from src.llm.factory import LLMFactory
        from src.store.factory import VectorStoreFactory
        
        # Create embeddings
        embeddings = EmbeddingsFactory.get_embeddings(settings.embeddings)
        
        # Create vector store
        store = VectorStoreFactory.get_store(
            store_type=settings.vectorstore.type,
            dimension=settings.embeddings.dimension,
            persist_directory=settings.vectorstore.persist_directory,
            collection_name=settings.vectorstore.collection_name,
        )
        
        # Create retrieval service
        global _retrieval_service
        _retrieval_service = RetrievalService(embeddings=embeddings, store=store)
        
        # Create LLM
        llm = LLMFactory.get_llm(settings.llm)
        
        # Create workflow
        _workflow = create_qna_graph(
            retrieval=_retrieval_service,
            llm=llm,
            config=settings.graph,
        )
        
        logger.info("LangGraph workflow initialized")
    
    return _workflow


@router.post(
    "",
    response_model=QueryResponse,
    summary="Process a question through the GraphRAG system",
    description="Submit a question to be processed through the LangGraph workflow.",
    responses={
        200: {
            "description": "Successful response with answer or refusal",
            "model": QueryResponse,
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Question cannot be empty"
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "type": "RAGError",
                            "message": "Failed to process query",
                            "details": "Connection refused"
                        }
                    }
                }
            },
        },
    },
)
async def query(request: QueryRequest, _: None = Depends(verify_api_key)) -> QueryResponse:
    """Process a question through the GraphRAG system.
    
    This endpoint accepts a question and processes it through the LangGraph
    workflow, which includes:
    1. Query classification (factual, procedural, or unsupported)
    2. Document retrieval from the vector store
    3. Answer generation using the LLM
    4. Grounding verification
    5. Retry/refuse decision based on confidence
    
    Args:
        request: The query request containing question, mode, and k.
        
    Returns:
        QueryResponse with answer, citations, confidence, and latency.
        
    Raises:
        HTTPException: If the request is invalid or processing fails.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Processing query",
        extra={
            "question": request.question[:100],
            "mode": request.mode,
            "k": request.k,
        }
    )
    
    try:
        # Get the workflow
        workflow = get_workflow()
        
        # Create initial state
        initial_state: GraphState = {
            "question": request.question,
            "query_type": "",
            "chunks": [],
            "search_results": [],
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "is_grounded": False,
            "retry_count": 0,
            "action": None,
            "refusal_reason": None,
            "error": None,
        }
        
        # Invoke the workflow
        result = workflow.invoke(initial_state)
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract response fields
        answer = result.get("answer")
        citations = result.get("citations", [])
        confidence = result.get("confidence", 0.0)
        refusal_reason = result.get("refusal_reason")
        error = result.get("error")
        
        # Handle workflow errors
        if error:
            logger.error(
                "Workflow error",
                extra={"error": error, "latency_ms": latency_ms}
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {error}"
            )
        
        # Log the result
        logger.info(
            "Query completed",
            extra={
                "has_answer": answer is not None,
                "confidence": confidence,
                "refused": refusal_reason is not None,
                "latency_ms": latency_ms,
            }
        )
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            refusal_reason=refusal_reason,
            latency_ms=latency_ms,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except RAGError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"RAGError during query: {e.message}",
            extra={"details": e.details, "latency_ms": latency_ms}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
        
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            f"Unexpected error during query: {str(e)}",
            extra={"latency_ms": latency_ms}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
