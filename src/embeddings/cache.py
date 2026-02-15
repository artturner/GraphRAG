"""Disk-based embedding cache implementation.

This module provides a persistent disk-based cache for embedding vectors,
allowing embeddings to be stored and retrieved across application restarts.

The cache uses a hash of the text and model name as the key, and stores
the embedding vectors as JSON files on disk.

Example:
    ```python
    from src.embeddings import DiskEmbeddingCache
    
    cache = DiskEmbeddingCache("./cache/embeddings")
    
    # Store an embedding
    cache.set("Hello world", "all-MiniLM-L6-v2", [0.1, 0.2, 0.3])
    
    # Retrieve an embedding
    vector = cache.get("Hello world", "all-MiniLM-L6-v2")
    print(vector)  # [0.1, 0.2, 0.3]
    
    # Clear the cache
    cache.clear()
    ```
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from src.embeddings.base import EmbeddingCache
from src.types import EmbeddingVector

logger = logging.getLogger(__name__)


class DiskEmbeddingCache(EmbeddingCache):
    """Disk-based persistent cache for embedding vectors.
    
    This class implements the EmbeddingCache protocol using disk storage.
    Embeddings are stored as JSON files in a configurable directory,
    with the filename derived from a hash of the text and model name.
    
    The cache is persistent across application restarts and provides
    efficient lookup of previously computed embeddings.
    
    Attributes:
        cache_dir: The directory where cache files are stored.
    
    Example:
        ```python
        # Create a cache in a specific directory
        cache = DiskEmbeddingCache("./cache/embeddings")
        
        # Store an embedding
        cache.set("Hello world", "all-MiniLM-L6-v2", [0.1, 0.2, 0.3])
        
        # Check if an embedding exists
        if cache.contains(cache._make_key("Hello world", "all-MiniLM-L6-v2")):
            vector = cache.get("Hello world", "all-MiniLM-L6-v2")
        
        # Clear all cached embeddings
        cache.clear()
        ```
    """
    
    def __init__(self, cache_dir: str | Path = "./cache/embeddings") -> None:
        """Initialize the disk-based embedding cache.
        
        Args:
            cache_dir: Directory path for storing cache files.
                Will be created if it doesn't exist.
        
        Example:
            ```python
            # Use default cache directory
            cache = DiskEmbeddingCache()
            
            # Use custom cache directory
            cache = DiskEmbeddingCache("/path/to/cache")
            ```
        """
        self._cache_dir = Path(cache_dir)
        self._ensure_cache_dir()
        logger.info(f"Initialized disk embedding cache at: {self._cache_dir}")
    
    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path.
        
        Returns:
            Path object for the cache directory.
        """
        return self._cache_dir
    
    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists.
        
        Creates the directory and any parent directories if they don't exist.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_key(self, text: str, model: str) -> str:
        """Generate a cache key from text and model name.
        
        The key is a SHA-256 hash of the concatenated text and model name,
        providing a unique and consistent identifier for each text-model pair.
        
        Args:
            text: The text that was embedded.
            model: The model name used for embedding.
            
        Returns:
            A hexadecimal string hash serving as the cache key.
        
        Example:
            ```python
            key = cache._make_key("Hello world", "all-MiniLM-L6-v2")
            print(key)  # "a1b2c3d4..." (64 character hex string)
            ```
        """
        # Combine text and model name for unique identification
        combined = f"{model}:{text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.
        
        Args:
            key: The cache key.
            
        Returns:
            Path object for the cache file.
        """
        return self._cache_dir / f"{key}.json"
    
    def get(self, key: str) -> EmbeddingVector | None:
        """Retrieve an embedding vector from the cache by key.
        
        This method implements the EmbeddingCache protocol. For convenience,
        use the get_by_text method to retrieve by text and model name.
        
        Args:
            key: The cache key (hash of text and model).
            
        Returns:
            The cached embedding vector, or None if not found.
        
        Example:
            ```python
            key = cache._make_key("Hello world", "all-MiniLM-L6-v2")
            vector = cache.get(key)
            ```
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss for key: {key[:16]}...")
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            vector = data.get("vector")
            if vector is not None:
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return vector
            return None
            
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
            return None
    
    def set(self, key: str, value: EmbeddingVector) -> None:
        """Store an embedding vector in the cache by key.
        
        This method implements the EmbeddingCache protocol. For convenience,
        use the set_by_text method to store by text and model name.
        
        Args:
            key: The cache key (hash of text and model).
            value: The embedding vector to cache.
        
        Example:
            ```python
            key = cache._make_key("Hello world", "all-MiniLM-L6-v2")
            cache.set(key, [0.1, 0.2, 0.3])
            ```
        """
        cache_path = self._get_cache_path(key)
        
        data = {
            "vector": value,
            "dimension": len(value),
        }
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.debug(f"Cached embedding for key: {key[:16]}...")
        except OSError as e:
            logger.error(f"Failed to write cache file {cache_path}: {e}")
    
    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: The cache key to check.
            
        Returns:
            True if the key exists in the cache, False otherwise.
        
        Example:
            ```python
            key = cache._make_key("Hello world", "all-MiniLM-L6-v2")
            if cache.contains(key):
                print("Embedding is cached")
            ```
        """
        return self._get_cache_path(key).exists()
    
    def clear(self) -> None:
        """Clear all entries from the cache.
        
        This removes all cache files from the cache directory.
        The directory itself is preserved.
        
        Example:
            ```python
            cache.clear()
            print("Cache cleared")
            ```
        """
        try:
            # Remove all files in the cache directory
            for cache_file in self._cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared cache directory: {self._cache_dir}")
        except OSError as e:
            logger.error(f"Failed to clear cache directory: {e}")
    
    def get_by_text(self, text: str, model: str) -> EmbeddingVector | None:
        """Retrieve an embedding vector by text and model name.
        
        This is a convenience method that generates the cache key
        automatically from the text and model name.
        
        Args:
            text: The text that was embedded.
            model: The model name used for embedding.
            
        Returns:
            The cached embedding vector, or None if not found.
        
        Example:
            ```python
            vector = cache.get_by_text("Hello world", "all-MiniLM-L6-v2")
            if vector:
                print(f"Found cached embedding with {len(vector)} dimensions")
            ```
        """
        key = self._make_key(text, model)
        return self.get(key)
    
    def set_by_text(self, text: str, model: str, vector: EmbeddingVector) -> None:
        """Store an embedding vector by text and model name.
        
        This is a convenience method that generates the cache key
        automatically from the text and model name.
        
        Args:
            text: The text that was embedded.
            model: The model name used for embedding.
            vector: The embedding vector to cache.
        
        Example:
            ```python
            cache.set_by_text("Hello world", "all-MiniLM-L6-v2", [0.1, 0.2, 0.3])
            ```
        """
        key = self._make_key(text, model)
        self.set(key, vector)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the cache.
        
        Returns:
            A dictionary with cache statistics including:
                - count: Number of cached embeddings
                - size_bytes: Total size of cache files in bytes
                - cache_dir: Path to the cache directory
        
        Example:
            ```python
            stats = cache.get_stats()
            print(f"Cached {stats['count']} embeddings, using {stats['size_bytes']} bytes")
            ```
        """
        cache_files = list(self._cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        
        return {
            "count": len(cache_files),
            "size_bytes": total_size,
            "cache_dir": str(self._cache_dir),
        }
    
    def __repr__(self) -> str:
        """Return a string representation of the cache.
        
        Returns:
            A string describing the cache instance.
        """
        stats = self.get_stats()
        return (
            f"DiskEmbeddingCache(cache_dir='{self._cache_dir}', "
            f"count={stats['count']}, size_bytes={stats['size_bytes']})"
        )