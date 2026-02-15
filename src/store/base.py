"""Base classes for vector store implementations.

This module defines the abstract base class for vector stores and the
SearchResult model for retrieval operations.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.types import Chunk, EmbeddingVector


class SearchResult(BaseModel):
    """Represents a search result from a vector store.
    
    This model captures a single result from a similarity search operation,
    including the chunk content, relevance score, and associated metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk.
        score: Similarity score between 0.0 and 1.0 (higher is more similar).
        chunk: The chunk object containing the text content.
        metadata: Additional metadata associated with the result.
    
    Example:
        ```python
        result = SearchResult(
            chunk_id="chunk-001",
            score=0.95,
            chunk=chunk_object,
            metadata={"source": "document.txt"}
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between 0.0 and 1.0"
    )
    chunk: Chunk = Field(..., description="The chunk object containing text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the result"
    )


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations.
    
    This class defines the interface for vector stores that persist and
    retrieve embeddings. All vector store implementations must implement
    this interface to ensure consistent behavior across the system.
    
    The class provides methods for adding embeddings, performing similarity
    search, deleting entries, and managing the store's contents.
    
    Example:
        ```python
        class MyStore(BaseVectorStore):
            def __init__(self):
                self._store: dict[str, tuple[Chunk, EmbeddingVector]] = {}
            
            def add_embeddings(
                self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
            ) -> int:
                for chunk, embedding in zip(chunks, embeddings):
                    self._store[chunk.id] = (chunk, embedding)
                return len(chunks)
            
            def similarity_search(
                self, query_embedding: EmbeddingVector, k: int
            ) -> list[SearchResult]:
                # Implementation using cosine similarity
                return []
            
            def delete(self, chunk_ids: list[str]) -> int:
                count = 0
                for cid in chunk_ids:
                    if cid in self._store:
                        del self._store[cid]
                        count += 1
                return count
            
            def count(self) -> int:
                return len(self._store)
            
            def clear(self) -> None:
                self._store.clear()
        ```
    """
    
    @abstractmethod
    def add_embeddings(
        self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
    ) -> int:
        """Add chunks and their embeddings to the vector store.
        
        This method stores the provided chunks along with their pre-computed
        embedding vectors for later retrieval via similarity search.
        
        Args:
            chunks: A list of Chunk objects to store.
            embeddings: A list of embedding vectors corresponding to each chunk.
                       Must be the same length as chunks.
        
        Returns:
            The number of chunks successfully added to the store.
        
        Raises:
            ValueError: If chunks and embeddings have different lengths.
            VectorStoreError: If the operation fails.
        
        Example:
            ```python
            store = MyStore()
            chunks = [chunk1, chunk2]
            embeddings = [[0.1, 0.2], [0.3, 0.4]]
            count = store.add_embeddings(chunks, embeddings)
            print(count)  # 2
            ```
        """
        ...
    
    @abstractmethod
    def similarity_search(
        self, query_embedding: EmbeddingVector, k: int
    ) -> list[SearchResult]:
        """Find the most similar chunks to a query embedding.
        
        This method performs a similarity search using the provided query
        embedding and returns the top-k most similar results.
        
        Args:
            query_embedding: The embedding vector of the query.
            k: The maximum number of results to return.
        
        Returns:
            A list of SearchResult objects sorted by similarity score
            (highest first), with at most k results.
        
        Raises:
            VectorStoreError: If the search operation fails.
        
        Example:
            ```python
            store = MyStore()
            query_vec = [0.1, 0.2, 0.3]
            results = store.similarity_search(query_vec, k=5)
            for result in results:
                print(f"{result.chunk_id}: {result.score}")
            ```
        """
        ...
    
    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks from the vector store by their IDs.
        
        This method removes the specified chunks and their associated
        embeddings from the store.
        
        Args:
            chunk_ids: A list of chunk IDs to delete.
        
        Returns:
            The number of chunks successfully deleted.
        
        Raises:
            VectorStoreError: If the delete operation fails.
        
        Example:
            ```python
            store = MyStore()
            deleted = store.delete(["chunk-001", "chunk-002"])
            print(f"Deleted {deleted} chunks")
            ```
        """
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks in the vector store.
        
        Returns:
            The total number of chunks currently stored.
        
        Example:
            ```python
            store = MyStore()
            print(f"Store contains {store.count()} chunks")
            ```
        """
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks and embeddings from the vector store.
        
        This method completely clears the store, removing all stored
        chunks and their associated embeddings.
        
        Example:
            ```python
            store = MyStore()
            store.clear()
            assert store.count() == 0
            ```
        """
        ...