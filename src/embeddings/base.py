"""Base classes and protocols for embedding providers.

This module defines the abstract base class for embedding providers and
the protocol for caching implementations.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from src.types import EmbeddingVector


@runtime_checkable
class EmbeddingCache(Protocol):
    """Protocol for embedding cache implementations.
    
    This protocol defines the interface for caching embedding vectors
    to avoid redundant API calls and improve performance.
    
    Implementations can use various backends such as in-memory dictionaries,
    Redis, or file-based storage.
    
    Example:
        ```python
        class InMemoryCache:
            def __init__(self):
                self._cache: dict[str, EmbeddingVector] = {}
            
            def get(self, key: str) -> EmbeddingVector | None:
                return self._cache.get(key)
            
            def set(self, key: str, value: EmbeddingVector) -> None:
                self._cache[key] = value
            
            def contains(self, key: str) -> bool:
                return key in self._cache
            
            def clear(self) -> None:
                self._cache.clear()
        ```
    """
    
    def get(self, key: str) -> EmbeddingVector | None:
        """Retrieve an embedding vector from the cache.
        
        Args:
            key: The cache key (typically a hash of the text).
            
        Returns:
            The cached embedding vector, or None if not found.
        """
        ...
    
    def set(self, key: str, value: EmbeddingVector) -> None:
        """Store an embedding vector in the cache.
        
        Args:
            key: The cache key (typically a hash of the text).
            value: The embedding vector to cache.
        """
        ...
    
    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: The cache key to check.
            
        Returns:
            True if the key exists in the cache, False otherwise.
        """
        ...
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        ...


class BaseEmbeddings(ABC):
    """Abstract base class for embedding providers.
    
    This class defines the interface for embedding providers that convert
    text into vector representations. All embedding providers must implement
    this interface to ensure consistent behavior across the system.
    
    The class provides methods for embedding both documents (batch) and
    queries (single), as well as a dimension property that indicates the
    size of the embedding vectors.
    
    Attributes:
        dimension: The dimensionality of the embedding vectors produced.
    
    Example:
        ```python
        class MyEmbeddings(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
        ```
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.
        
        This property indicates the size of the embedding vectors produced
        by this provider. Different models produce vectors of different sizes.
        
        Returns:
            The number of dimensions in each embedding vector.
        
        Example:
            ```python
            embeddings = MyEmbeddings()
            print(embeddings.dimension)  # 768
            ```
        """
        ...
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a list of documents into vector representations.
        
        This method is optimized for batch processing of multiple documents.
        It's typically used during the ingestion phase to create embeddings
        for document chunks that will be stored in the vector database.
        
        Args:
            texts: A list of text strings to embed.
            
        Returns:
            A list of embedding vectors, one for each input text.
            The order of outputs matches the order of inputs.
        
        Raises:
            EmbeddingError: If the embedding operation fails.
            ValueError: If texts is empty or contains invalid values.
        
        Example:
            ```python
            embeddings = MyEmbeddings()
            texts = ["Hello world", "Goodbye world"]
            vectors = embeddings.embed_documents(texts)
            print(len(vectors))  # 2
            print(len(vectors[0]))  # 768 (dimension)
            ```
        """
        ...
    
    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single query into a vector representation.
        
        This method is optimized for embedding a single query text.
        It's typically used during retrieval to create an embedding
        for the user's question that will be used to find similar documents.
        
        Some embedding providers use different models or parameters for
        queries vs documents, so this method is kept separate from
        embed_documents.
        
        Args:
            text: The query text to embed.
            
        Returns:
            An embedding vector representing the query.
        
        Raises:
            EmbeddingError: If the embedding operation fails.
            ValueError: If text is empty or invalid.
        
        Example:
            ```python
            embeddings = MyEmbeddings()
            query = "What is machine learning?"
            vector = embeddings.embed_query(query)
            print(len(vector))  # 768 (dimension)
            ```
        """
        ...