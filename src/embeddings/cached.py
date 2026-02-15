"""Cached embedding wrapper for transparent caching.

This module provides a wrapper class that adds transparent caching to any
embedding provider. It intercepts embedding requests and returns cached
results when available, only calling the underlying provider when necessary.

Example:
    ```python
    from src.embeddings import LocalEmbeddings, CachedEmbeddings, DiskEmbeddingCache
    
    # Create the underlying embedding provider
    embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create a cache
    cache = DiskEmbeddingCache("./cache/embeddings")
    
    # Wrap with caching
    cached_embeddings = CachedEmbeddings(embeddings, cache)
    
    # First call computes and caches
    vector1 = cached_embeddings.embed_query("Hello world")
    
    # Second call returns cached result
    vector2 = cached_embeddings.embed_query("Hello world")
    # vector1 == vector2
    ```
"""

import logging
from typing import Any

from src.embeddings.base import BaseEmbeddings, EmbeddingCache
from src.types import EmbeddingVector

logger = logging.getLogger(__name__)


class CachedEmbeddings(BaseEmbeddings):
    """Wrapper that adds transparent caching to any embedding provider.
    
    This class wraps any BaseEmbeddings implementation and adds caching
    functionality. It intercepts embedding requests and returns cached
    results when available, only calling the underlying provider when
    the result is not in the cache.
    
    The wrapper is transparent - it implements the same BaseEmbeddings
    interface and can be used anywhere a BaseEmbeddings is expected.
    
    Attributes:
        _embeddings: The underlying embedding provider.
        _cache: The cache implementation.
        _model_name: The model name used for cache keys.
    
    Example:
        ```python
        from src.embeddings import OpenAIEmbeddings, CachedEmbeddings, DiskEmbeddingCache
        
        # Create OpenAI embeddings
        openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create cache
        cache = DiskEmbeddingCache("./cache/openai_embeddings")
        
        # Wrap with caching
        cached = CachedEmbeddings(openai_emb, cache, model_name="text-embedding-3-small")
        
        # Use transparently - caching happens automatically
        vectors = cached.embed_documents(["Hello", "World"])
        ```
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        cache: EmbeddingCache,
        model_name: str | None = None,
    ) -> None:
        """Initialize the cached embeddings wrapper.
        
        Args:
            embeddings: The underlying embedding provider to wrap.
            cache: The cache implementation to use.
            model_name: Optional model name for cache keys. If not provided,
                a default name will be used. This is important for ensuring
                cache keys are unique per model.
        
        Example:
            ```python
            embeddings = LocalEmbeddings()
            cache = DiskEmbeddingCache()
            cached = CachedEmbeddings(embeddings, cache, model_name="all-MiniLM-L6-v2")
            ```
        """
        self._embeddings = embeddings
        self._cache = cache
        self._model_name = model_name or "default_model"
        
        logger.info(
            f"Initialized CachedEmbeddings wrapper for {type(embeddings).__name__} "
            f"with model '{self._model_name}'"
        )
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.
        
        This delegates to the underlying embedding provider.
        
        Returns:
            The number of dimensions in each embedding vector.
        """
        return self._embeddings.dimension
    
    @property
    def underlying_embeddings(self) -> BaseEmbeddings:
        """Return the underlying embedding provider.
        
        Returns:
            The wrapped BaseEmbeddings instance.
        """
        return self._embeddings
    
    @property
    def cache(self) -> EmbeddingCache:
        """Return the cache implementation.
        
        Returns:
            The EmbeddingCache instance used for caching.
        """
        return self._cache
    
    @property
    def model_name(self) -> str:
        """Return the model name used for cache keys.
        
        Returns:
            The model name string.
        """
        return self._model_name
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text.
        
        Args:
            text: The text to generate a key for.
            
        Returns:
            A cache key string.
        """
        # Use the cache's key generation if it's a DiskEmbeddingCache
        if hasattr(self._cache, "_make_key"):
            return self._cache._make_key(text, self._model_name)
        # Fallback to simple hash-based key
        import hashlib
        combined = f"{self._model_name}:{text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single query with caching.
        
        This method first checks the cache for the embedding. If found,
        it returns the cached result. Otherwise, it calls the underlying
        provider, caches the result, and returns it.
        
        Args:
            text: The query text to embed.
            
        Returns:
            An embedding vector representing the query.
        
        Example:
            ```python
            vector = cached_embeddings.embed_query("What is machine learning?")
            ```
        """
        # Check cache first
        if hasattr(self._cache, "get_by_text"):
            cached_result = self._cache.get_by_text(text, self._model_name)
        else:
            key = self._get_cache_key(text)
            cached_result = self._cache.get(key)
        
        if cached_result is not None:
            logger.debug(f"Cache hit for query: {text[:50]}...")
            return cached_result
        
        # Cache miss - compute embedding
        logger.debug(f"Cache miss for query: {text[:50]}...")
        vector = self._embeddings.embed_query(text)
        
        # Store in cache
        if hasattr(self._cache, "set_by_text"):
            self._cache.set_by_text(text, self._model_name, vector)
        else:
            key = self._get_cache_key(text)
            self._cache.set(key, vector)
        
        return vector
    
    def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a list of documents with caching.
        
        This method checks the cache for each document and only computes
        embeddings for documents not in the cache. This can significantly
        reduce API calls for repeated documents.
        
        Args:
            texts: A list of text strings to embed.
            
        Returns:
            A list of embedding vectors, one for each input text.
        
        Example:
            ```python
            texts = ["Document 1", "Document 2", "Document 3"]
            vectors = cached_embeddings.embed_documents(texts)
            ```
        """
        results: list[EmbeddingVector | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if hasattr(self._cache, "get_by_text"):
                cached_result = self._cache.get_by_text(text, self._model_name)
            else:
                key = self._get_cache_key(text)
                cached_result = self._cache.get(key)
            
            if cached_result is not None:
                results[i] = cached_result
                logger.debug(f"Cache hit for document {i}: {text[:50]}...")
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Compute embeddings for uncached texts
        if uncached_texts:
            logger.debug(
                f"Cache miss for {len(uncached_texts)} of {len(texts)} documents"
            )
            new_vectors = self._embeddings.embed_documents(uncached_texts)
            
            # Store new embeddings in cache and update results
            for idx, text, vector in zip(uncached_indices, uncached_texts, new_vectors):
                if hasattr(self._cache, "set_by_text"):
                    self._cache.set_by_text(text, self._model_name, vector)
                else:
                    key = self._get_cache_key(text)
                    self._cache.set(key, vector)
                results[idx] = vector
        
        # All results should be populated now
        return [r for r in results if r is not None]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache.
        
        This method clears all entries from the underlying cache.
        
        Example:
            ```python
            cached_embeddings.clear_cache()
            ```
        """
        self._cache.clear()
        logger.info(f"Cleared cache for model: {self._model_name}")
    
    def __repr__(self) -> str:
        """Return a string representation of the cached embeddings.
        
        Returns:
            A string describing the cached embeddings instance.
        """
        return (
            f"CachedEmbeddings("
            f"embeddings={type(self._embeddings).__name__}, "
            f"model='{self._model_name}', "
            f"dimension={self.dimension})"
        )