"""Retrieval service with citation extraction and reranking.

This module provides the RetrievalService class for indexing and
retrieving documents using embeddings and vector storage, the
CitationBuilder class for extracting and formatting citations,
and reranker implementations for improving retrieval quality.
"""

from src.retrieval.citations import CitationBuilder
from src.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    IdentityReranker,
)
from src.retrieval.service import RetrievalService

__all__ = [
    "BaseReranker",
    "CitationBuilder",
    "CrossEncoderReranker",
    "IdentityReranker",
    "RetrievalService",
]
