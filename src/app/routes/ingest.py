"""Ingest endpoint for the GraphRAG API.

This module provides the POST /ingest endpoint that triggers the document
ingestion pipeline for a specified corpus.
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.app.schemas import IngestRequest, IngestResponse
from src.config import CorpusConfig, settings
from src.connectors.factory import ConnectorFactory
from src.exceptions import RAGError
from src.ingestion.chunking import FixedSizeChunker
from src.ingestion.cleaning import TextCleaner
from src.ingestion.pipeline import IngestionPipeline
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["documents"])

# Track async ingestion status
_ingestion_status: dict[str, IngestResponse] = {}


def get_ingestion_pipeline(corpus: str) -> IngestionPipeline:
    """Create an ingestion pipeline for the specified corpus.
    
    Args:
        corpus: The name of the corpus to ingest.
        
    Returns:
        Configured IngestionPipeline instance.
        
    Raises:
        HTTPException: If the corpus configuration is invalid.
    """
    try:
        # Create corpus config from settings or use defaults
        corpus_config = CorpusConfig(
            name=corpus,
            path=settings.corpus.path,
            connector_type=settings.corpus.connector_type,
        )
        
        # Create connector
        connector = ConnectorFactory.get_connector(corpus_config)
        
        # Create text cleaner with default settings
        cleaner = TextCleaner()
        
        # Create chunker with default settings
        chunker = FixedSizeChunker(
            chunk_size=500,
            overlap=50,
        )
        
        # Create and return pipeline
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        logger.info(
            f"Created ingestion pipeline for corpus: {corpus}",
            extra={
                "connector_type": corpus_config.connector_type,
                "path": corpus_config.path,
            }
        )
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to create ingestion pipeline: {str(e)}")
        raise


async def run_async_ingestion(corpus: str) -> None:
    """Run ingestion asynchronously in the background.
    
    Args:
        corpus: The name of the corpus to ingest.
    """
    _ingestion_status[corpus] = IngestResponse(
        status="in_progress",
        documents_count=0,
        chunks_count=0,
        errors=[],
    )
    
    try:
        pipeline = get_ingestion_pipeline(corpus)
        
        # Run the blocking pipeline in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pipeline.run)
        
        _ingestion_status[corpus] = IngestResponse(
            status="completed",
            documents_count=result.documents_count,
            chunks_count=result.chunks_count,
            errors=result.errors,
        )
        
        logger.info(
            f"Async ingestion completed for corpus: {corpus}",
            extra={
                "documents_count": result.documents_count,
                "chunks_count": result.chunks_count,
            }
        )
        
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        logger.error(f"Async ingestion failed for corpus: {corpus}", extra={"error": str(e)})
        
        _ingestion_status[corpus] = IngestResponse(
            status="failed",
            documents_count=0,
            chunks_count=0,
            errors=[error_msg],
        )


@router.post(
    "",
    response_model=IngestResponse,
    summary="Trigger document ingestion",
    description="Start the ingestion pipeline for the specified corpus.",
    responses={
        200: {
            "description": "Ingestion completed or started successfully",
            "model": IngestResponse,
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Corpus name cannot be empty"
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
                            "message": "Failed to run ingestion pipeline",
                            "details": "Connection refused"
                        }
                    }
                }
            },
        },
    },
)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Trigger document ingestion for the specified corpus.
    
    This endpoint starts the ingestion pipeline which:
    1. Loads documents from the configured source
    2. Cleans and normalizes the text content
    3. Chunks documents into smaller pieces
    4. Returns statistics about the ingestion
    
    Args:
        request: The ingest request containing corpus name and async flag.
        
    Returns:
        IngestResponse with status, counts, and any errors.
        
    Raises:
        HTTPException: If the request is invalid or ingestion fails.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Starting ingestion",
        extra={
            "corpus": request.corpus,
            "async_ingest": request.async_ingest,
        }
    )
    
    try:
        if request.async_ingest:
            # Start async ingestion in background
            asyncio.create_task(run_async_ingestion(request.corpus))
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Async ingestion started",
                extra={"corpus": request.corpus, "latency_ms": latency_ms}
            )
            
            return IngestResponse(
                status="in_progress",
                documents_count=0,
                chunks_count=0,
                errors=[],
            )
        
        # Run synchronous ingestion
        pipeline = get_ingestion_pipeline(request.corpus)
        result = pipeline.run()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Determine status based on errors
        status_value = "completed" if not result.errors else "failed"
        
        logger.info(
            "Ingestion completed",
            extra={
                "corpus": request.corpus,
                "documents_count": result.documents_count,
                "chunks_count": result.chunks_count,
                "errors_count": len(result.errors),
                "latency_ms": latency_ms,
            }
        )
        
        return IngestResponse(
            status=status_value,
            documents_count=result.documents_count,
            chunks_count=result.chunks_count,
            errors=result.errors,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except RAGError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"RAGError during ingestion: {e.message}",
            extra={"details": e.details, "latency_ms": latency_ms}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
        
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            f"Unexpected error during ingestion: {str(e)}",
            extra={"latency_ms": latency_ms}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.get(
    "/status/{corpus}",
    response_model=IngestResponse,
    summary="Get ingestion status",
    description="Check the status of an async ingestion operation.",
    responses={
        200: {
            "description": "Current ingestion status",
            "model": IngestResponse,
        },
        404: {
            "description": "No ingestion found for corpus",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "No ingestion found for corpus: my_corpus"
                    }
                }
            },
        },
    },
)
async def get_status(corpus: str) -> IngestResponse:
    """Get the status of an async ingestion operation.
    
    Args:
        corpus: The name of the corpus to check status for.
        
    Returns:
        IngestResponse with current status.
        
    Raises:
        HTTPException: If no ingestion is found for the corpus.
    """
    if corpus not in _ingestion_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingestion found for corpus: {corpus}",
        )
    
    return _ingestion_status[corpus]
