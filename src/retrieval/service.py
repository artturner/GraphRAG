"""Retrieval service that combines embeddings and vector store.

This module provides the RetrievalService class which orchestrates
embedding generation and vector storage for document retrieval.
"""

import logging
from typing import Any

from src.embeddings.base import BaseEmbeddings
from src.exceptions import RetrievalError
from src.store.base import BaseVectorStore, SearchResult
from src.types import Chunk

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for indexing and retrieving documents using embeddings.
    
    This service combines an embedding provider with a vector store to
    provide a unified interface for document indexing and retrieval.
    
    The service handles:
    - Converting document chunks to embeddings
    - Storing embeddings in the vector store
    - Performing similarity searches
    - Filtering results by relevance threshold
    
    Attributes:
        embeddings: The embedding provider for generating vectors.
        store: The vector store for persisting and searching embeddings.
    
    Example:
        ```python
        from src.retrieval import RetrievalService
        from src.embeddings import LocalEmbeddings
        from src.store import FAISSVectorStore
        
        embeddings = LocalEmbeddings()
        store = FAISSVectorStore(dimension=embeddings.dimension)
        retrieval = RetrievalService(embeddings, store)
        
        # Index documents
        chunks = [chunk1, chunk2, chunk3]
        count = retrieval.index_documents(chunks)
        
        # Search for relevant documents
        results = retrieval.search("What is federalism?", k=5)
        for result in results:
            print(f"{result.chunk_id}: {result.score:.3f}")
        ```
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        store: BaseVectorStore,
    ) -> None:
        """Initialize the retrieval service.
        
        Args:
            embeddings: The embedding provider for generating vectors.
            store: The vector store for persisting and searching embeddings.
            
        Raises:
            ValueError: If embeddings and store have incompatible dimensions.
        
        Example:
            ```python
            embeddings = LocalEmbeddings()
            store = FAISSVectorStore(dimension=embeddings.dimension)
            retrieval = RetrievalService(embeddings, store)
            ```
        """
        self._embeddings = embeddings
        self._store = store
        
        logger.debug(
            "Initialized RetrievalService with embeddings dimension=%d",
            embeddings.dimension,
        )
    
    @property
    def embeddings(self) -> BaseEmbeddings:
        """Return the embedding provider."""
        return self._embeddings
    
    @property
    def store(self) -> BaseVectorStore:
        """Return the vector store."""
        return self._store
    
    def index_documents(self, chunks: list[Chunk]) -> int:
        """Index document chunks by generating and storing embeddings.
        
        This method:
        1. Extracts text content from each chunk
        2. Generates embeddings for all chunks in batch
        3. Stores the chunks and embeddings in the vector store
        
        Args:
            chunks: A list of Chunk objects to index.
            
        Returns:
            The number of chunks successfully indexed.
            
        Raises:
            RetrievalError: If embedding generation or storage fails.
            ValueError: If chunks is empty.
        
        Example:
            ```python
            chunks = [
                Chunk(
                    id="chunk-001",
                    document_id="doc-001",
                    content="Federalism is a system of government...",
                    start_idx=0,
                    end_idx=100,
                ),
            ]
            count = retrieval.index_documents(chunks)
            print(f"Indexed {count} chunks")
            ```
        """
        if not chunks:
            raise ValueError("Cannot index empty list of chunks")
        
        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            logger.info("Generating embeddings for %d chunks", len(chunks))
            
            # Generate embeddings in batch
            embedding_vectors = self._embeddings.embed_documents(texts)
            
            # Store chunks with their embeddings
            count = self._store.add_embeddings(chunks, embedding_vectors)
            
            logger.info("Successfully indexed %d chunks", count)
            
            return count
            
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error("Failed to index documents: %s", str(e))
            raise RetrievalError(
                "Failed to index documents",
                details=str(e),
            ) from e
    
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for documents similar to the query.
        
        This method:
        1. Generates an embedding for the query
        2. Performs a similarity search in the vector store
        3. Returns the top-k most similar results
        
        Args:
            query: The search query text.
            k: The maximum number of results to return. Defaults to 5.
            
        Returns:
            A list of SearchResult objects sorted by similarity score
            (highest first), with at most k results.
            
        Raises:
            RetrievalError: If the search operation fails.
            ValueError: If query is empty or k is not positive.
        
        Example:
            ```python
            results = retrieval.search("What is federalism?", k=5)
            for result in results:
                print(f"Score: {result.score:.3f}")
                print(f"Content: {result.chunk.content[:100]}...")
            ```
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if k <= 0:
            raise ValueError("k must be a positive integer")
        
        try:
            logger.debug("Searching for: %s (k=%d)", query[:50], k)
            
            # Generate embedding for the query
            query_embedding = self._embeddings.embed_query(query)
            
            # Perform similarity search
            results = self._store.similarity_search(query_embedding, k)
            
            logger.debug("Found %d results", len(results))
            
            return results
            
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error("Search failed: %s", str(e))
            raise RetrievalError(
                "Search operation failed",
                details=str(e),
            ) from e
    
    def search_with_threshold(
        self,
        query: str,
        k: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Search for documents with a minimum similarity threshold.
        
        This method performs a similarity search and filters the results
        to only include those with a score at or above the specified threshold.
        
        Args:
            query: The search query text.
            k: The maximum number of results to return before filtering.
            min_score: The minimum similarity score (0.0 to 1.0) required.
            
        Returns:
            A list of SearchResult objects with scores >= min_score,
            sorted by similarity score (highest first).
            
        Raises:
            RetrievalError: If the search operation fails.
            ValueError: If query is empty, k is not positive, or min_score
                       is not between 0.0 and 1.0.
        
        Example:
            ```python
            # Only return results with at least 80% similarity
            results = retrieval.search_with_threshold(
                "What is federalism?",
                k=10,
                min_score=0.8,
            )
            for result in results:
                print(f"Score: {result.score:.3f} (>= 0.8)")
            ```
        """
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        
        # Get all results from standard search
        all_results = self.search(query, k)
        
        # Filter by threshold
        filtered_results = [
            result for result in all_results
            if result.score >= min_score
        ]
        
        logger.debug(
            "Filtered %d results to %d with min_score=%.3f",
            len(all_results),
            len(filtered_results),
            min_score,
        )
        
        return filtered_results