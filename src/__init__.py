"""Grounded GraphRAG Tutor -- a citation-backed RAG service.

This package provides a content-agnostic RAG pipeline with LangGraph
orchestration that answers questions only when supported by sources,
returns citations, and refuses gracefully when evidence is insufficient.

Quick start::

    from src.config import settings
    from src.connectors import LocalConnector
    from src.embeddings import EmbeddingsFactory
    from src.store import VectorStoreFactory
    from src.retrieval import RetrievalService
    from src.llm import LLMFactory
    from src.graphs import create_qna_graph

Usage (logging)::

    from src import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"doc_id": "123"})
"""

__version__ = "0.1.0"

from src.logging_config import (
    clear_correlation_id,
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)

# Initialize logging on import
configure_logging()

# -- Public re-exports for convenience ------------------------------------

from src.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    DocumentMetadata,
    DocumentType,
    IngestResult,
    QueryResult,
)
from src.exceptions import (
    RAGError,
    ConfigurationError,
    ConnectorError,
    EmbeddingError,
    IngestionError,
    LLMError,
    RetrievalError,
    VectorStoreError,
)
from src.config import Settings, settings

__all__ = [
    # Version
    "__version__",
    # Logging
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "configure_logging",
    # Types
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "DocumentMetadata",
    "DocumentType",
    "IngestResult",
    "QueryResult",
    # Exceptions
    "RAGError",
    "ConfigurationError",
    "ConnectorError",
    "EmbeddingError",
    "IngestionError",
    "LLMError",
    "RetrievalError",
    "VectorStoreError",
    # Configuration
    "Settings",
    "settings",
]
