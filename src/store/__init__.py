"""Vector store adapters for storing and retrieving embeddings.

This module provides the base class for vector store implementations,
the SearchResult model for retrieval operations, and a factory for
creating vector store instances.
"""

from src.store.base import BaseVectorStore, SearchResult
from src.store.faiss_store import FAISSVectorStore
from src.store.chroma_store import ChromaVectorStore
from src.store.factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "SearchResult",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "VectorStoreFactory",
]
