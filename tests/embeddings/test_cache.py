"""Tests for the DiskEmbeddingCache class.

This module tests the disk-based embedding cache functionality including
cache hits, misses, persistence, and key generation.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.embeddings import DiskEmbeddingCache
from src.embeddings.cache import DiskEmbeddingCache as CacheClass
from src.types import EmbeddingVector


class TestDiskEmbeddingCache:
    """Test suite for DiskEmbeddingCache class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance with temporary directory."""
        return DiskEmbeddingCache(temp_cache_dir)
    
    # ==================== Key Generation Tests ====================
    
    def test_make_key_generates_consistent_hash(self, cache):
        """Test that _make_key generates consistent hashes for same input."""
        key1 = cache._make_key("Hello world", "all-MiniLM-L6-v2")
        key2 = cache._make_key("Hello world", "all-MiniLM-L6-v2")
        
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA-256 produces 64 hex characters
    
    def test_make_key_different_for_different_text(self, cache):
        """Test that different text produces different keys."""
        key1 = cache._make_key("Hello world", "all-MiniLM-L6-v2")
        key2 = cache._make_key("Goodbye world", "all-MiniLM-L6-v2")
        
        assert key1 != key2
    
    def test_make_key_different_for_different_model(self, cache):
        """Test that different model names produce different keys."""
        key1 = cache._make_key("Hello world", "all-MiniLM-L6-v2")
        key2 = cache._make_key("Hello world", "text-embedding-3-small")
        
        assert key1 != key2
    
    def test_make_key_includes_model_in_hash(self, cache):
        """Test that model name is included in hash generation."""
        # Same text with different models should produce different keys
        text = "Test text"
        key_local = cache._make_key(text, "all-MiniLM-L6-v2")
        key_openai = cache._make_key(text, "text-embedding-3-small")
        
        assert key_local != key_openai
    
    # ==================== Cache Hit/Miss Tests ====================
    
    def test_cache_miss_returns_none(self, cache):
        """Test that cache miss returns None."""
        result = cache.get("nonexistent_key")
        
        assert result is None
    
    def test_cache_set_and_get(self, cache):
        """Test basic cache set and get operations."""
        key = "test_key"
        vector: EmbeddingVector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        cache.set(key, vector)
        result = cache.get(key)
        
        assert result == vector
    
    def test_cache_hit_after_set(self, cache):
        """Test that cache returns stored value on hit."""
        key = cache._make_key("test text", "test-model")
        vector: EmbeddingVector = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        cache.set(key, vector)
        result = cache.get(key)
        
        assert result is not None
        assert result == vector
    
    def test_cache_miss_for_different_key(self, cache):
        """Test that get with different key returns None."""
        key1 = "key1"
        key2 = "key2"
        vector: EmbeddingVector = [0.1, 0.2, 0.3]
        
        cache.set(key1, vector)
        result = cache.get(key2)
        
        assert result is None
    
    # ==================== Convenience Method Tests ====================
    
    def test_get_by_text(self, cache):
        """Test get_by_text convenience method."""
        text = "Hello world"
        model = "all-MiniLM-L6-v2"
        vector: EmbeddingVector = [0.1, 0.2, 0.3]
        
        cache.set_by_text(text, model, vector)
        result = cache.get_by_text(text, model)
        
        assert result == vector
    
    def test_set_by_text(self, cache):
        """Test set_by_text convenience method."""
        text = "Test document"
        model = "test-model"
        vector: EmbeddingVector = [0.5, 0.6, 0.7]
        
        cache.set_by_text(text, model, vector)
        
        # Verify it can be retrieved
        result = cache.get_by_text(text, model)
        assert result == vector
    
    def test_get_by_text_miss(self, cache):
        """Test get_by_text returns None for missing entry."""
        result = cache.get_by_text("nonexistent", "model")
        
        assert result is None
    
    # ==================== Contains Tests ====================
    
    def test_contains_returns_true_for_existing_key(self, cache):
        """Test contains returns True for existing key."""
        key = "test_key"
        vector: EmbeddingVector = [0.1, 0.2]
        
        cache.set(key, vector)
        
        assert cache.contains(key) is True
    
    def test_contains_returns_false_for_missing_key(self, cache):
        """Test contains returns False for missing key."""
        assert cache.contains("nonexistent_key") is False
    
    # ==================== Clear Tests ====================
    
    def test_clear_removes_all_entries(self, cache):
        """Test that clear removes all cache entries."""
        cache.set("key1", [0.1, 0.2])
        cache.set("key2", [0.3, 0.4])
        cache.set("key3", [0.5, 0.6])
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None
    
    def test_clear_preserves_directory(self, cache, temp_cache_dir):
        """Test that clear preserves the cache directory itself."""
        cache.set("key1", [0.1, 0.2])
        
        cache.clear()
        
        assert os.path.exists(temp_cache_dir)
    
    # ==================== Persistence Tests ====================
    
    def test_cache_persists_to_disk(self, cache, temp_cache_dir):
        """Test that cache entries are persisted to disk."""
        key = "persist_test"
        vector: EmbeddingVector = [0.1, 0.2, 0.3]
        
        cache.set(key, vector)
        
        # Check that file was created
        cache_path = cache._get_cache_path(key)
        assert cache_path.exists()
        
        # Check file contents
        with open(cache_path, "r") as f:
            data = json.load(f)
        
        assert data["vector"] == vector
        assert data["dimension"] == len(vector)
    
    def test_cache_loads_from_disk(self, temp_cache_dir):
        """Test that cache entries can be loaded from disk."""
        key = "load_test"
        vector: EmbeddingVector = [0.4, 0.5, 0.6]
        
        # Create cache file manually
        cache_path = Path(temp_cache_dir) / f"{key}.json"
        data = {"vector": vector, "dimension": len(vector)}
        with open(cache_path, "w") as f:
            json.dump(data, f)
        
        # Create new cache instance and load
        new_cache = DiskEmbeddingCache(temp_cache_dir)
        result = new_cache.get(key)
        
        assert result == vector
    
    def test_cache_survives_instance_recreation(self, temp_cache_dir):
        """Test that cache survives creating new instance."""
        text = "Persistence test"
        model = "test-model"
        vector: EmbeddingVector = [0.7, 0.8, 0.9]
        
        # Create first cache instance and store
        cache1 = DiskEmbeddingCache(temp_cache_dir)
        cache1.set_by_text(text, model, vector)
        
        # Create second cache instance and retrieve
        cache2 = DiskEmbeddingCache(temp_cache_dir)
        result = cache2.get_by_text(text, model)
        
        assert result == vector
    
    # ==================== Statistics Tests ====================
    
    def test_get_stats_empty_cache(self, cache):
        """Test get_stats returns correct stats for empty cache."""
        stats = cache.get_stats()
        
        assert stats["count"] == 0
        assert stats["size_bytes"] == 0
        assert stats["cache_dir"] is not None
    
    def test_get_stats_with_entries(self, cache):
        """Test get_stats returns correct stats with entries."""
        cache.set("key1", [0.1] * 10)
        cache.set("key2", [0.2] * 20)
        
        stats = cache.get_stats()
        
        assert stats["count"] == 2
        assert stats["size_bytes"] > 0
    
    # ==================== Repr Tests ====================
    
    def test_repr(self, cache):
        """Test __repr__ returns meaningful string."""
        repr_str = repr(cache)
        
        assert "DiskEmbeddingCache" in repr_str
        assert "count=" in repr_str
        assert "size_bytes=" in repr_str
    
    # ==================== Error Handling Tests ====================
    
    def test_get_handles_corrupted_file(self, cache, temp_cache_dir):
        """Test that get handles corrupted JSON file gracefully."""
        key = "corrupted_key"
        cache_path = cache._get_cache_path(key)
        
        # Write invalid JSON
        with open(cache_path, "w") as f:
            f.write("not valid json {{{")
        
        result = cache.get(key)
        
        assert result is None
    
    def test_get_handles_missing_vector_in_file(self, cache, temp_cache_dir):
        """Test that get handles file without vector field."""
        key = "no_vector_key"
        cache_path = cache._get_cache_path(key)
        
        # Write JSON without vector field
        with open(cache_path, "w") as f:
            json.dump({"dimension": 10}, f)
        
        result = cache.get(key)
        
        assert result is None
    
    # ==================== Directory Creation Tests ====================
    
    def test_creates_cache_directory_if_not_exists(self):
        """Test that cache creates directory if it doesn't exist."""
        temp_base = tempfile.mkdtemp()
        cache_dir = os.path.join(temp_base, "nested", "cache", "dir")
        
        try:
            cache = DiskEmbeddingCache(cache_dir)
            
            assert os.path.exists(cache_dir)
            assert os.path.isdir(cache_dir)
        finally:
            shutil.rmtree(temp_base)
    
    # ==================== Large Vector Tests ====================
    
    def test_large_vector_storage(self, cache):
        """Test storing and retrieving large embedding vectors."""
        # Simulate a typical embedding dimension (e.g., 1536 for OpenAI)
        large_vector: EmbeddingVector = [0.1 * (i % 10) for i in range(1536)]
        
        cache.set("large_key", large_vector)
        result = cache.get("large_key")
        
        assert result == large_vector
        assert len(result) == 1536
    
    # ==================== Special Character Tests ====================
    
    def test_special_characters_in_text(self, cache):
        """Test caching text with special characters."""
        text = "Hello! @#$%^&*() \n\t\r World"
        model = "test-model"
        vector: EmbeddingVector = [0.1, 0.2, 0.3]
        
        cache.set_by_text(text, model, vector)
        result = cache.get_by_text(text, model)
        
        assert result == vector
    
    def test_unicode_text(self, cache):
        """Test caching text with unicode characters."""
        text = "Hello world in Chinese:  world in Arabic:  world in emoji: "
        model = "test-model"
        vector: EmbeddingVector = [0.4, 0.5, 0.6]
        
        cache.set_by_text(text, model, vector)
        result = cache.get_by_text(text, model)
        
        assert result == vector
    
    # ==================== Empty/Edge Case Tests ====================
    
    def test_empty_text(self, cache):
        """Test caching empty text."""
        text = ""
        model = "test-model"
        vector: EmbeddingVector = [0.1, 0.2]
        
        cache.set_by_text(text, model, vector)
        result = cache.get_by_text(text, model)
        
        assert result == vector
    
    def test_empty_vector(self, cache):
        """Test storing empty vector."""
        key = "empty_vector_key"
        vector: EmbeddingVector = []
        
        cache.set(key, vector)
        result = cache.get(key)
        
        assert result == []
    
    def test_single_dimension_vector(self, cache):
        """Test storing single-dimension vector."""
        key = "single_dim_key"
        vector: EmbeddingVector = [0.5]
        
        cache.set(key, vector)
        result = cache.get(key)
        
        assert result == [0.5]


class TestDiskEmbeddingCacheProtocol:
    """Test that DiskEmbeddingCache implements EmbeddingCache protocol."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance with temporary directory."""
        return DiskEmbeddingCache(temp_cache_dir)
    
    def test_has_get_method(self, cache):
        """Test that cache has get method."""
        assert hasattr(cache, 'get')
        assert callable(cache.get)
    
    def test_has_set_method(self, cache):
        """Test that cache has set method."""
        assert hasattr(cache, 'set')
        assert callable(cache.set)
    
    def test_has_contains_method(self, cache):
        """Test that cache has contains method."""
        assert hasattr(cache, 'contains')
        assert callable(cache.contains)
    
    def test_has_clear_method(self, cache):
        """Test that cache has clear method."""
        assert hasattr(cache, 'clear')
        assert callable(cache.clear)


class TestCacheKeyUniqueness:
    """Test cache key uniqueness across different scenarios."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance with temporary directory."""
        return DiskEmbeddingCache(temp_cache_dir)
    
    def test_different_models_same_text_different_keys(self, cache):
        """Test that same text with different models produces different cache entries."""
        text = "The quick brown fox"
        model1 = "model-a"
        model2 = "model-b"
        vector1: EmbeddingVector = [0.1, 0.2]
        vector2: EmbeddingVector = [0.3, 0.4]
        
        cache.set_by_text(text, model1, vector1)
        cache.set_by_text(text, model2, vector2)
        
        result1 = cache.get_by_text(text, model1)
        result2 = cache.get_by_text(text, model2)
        
        assert result1 == vector1
        assert result2 == vector2
        assert result1 != result2
    
    def test_similar_texts_different_keys(self, cache):
        """Test that similar texts produce different keys."""
        text1 = "Hello world"
        text2 = "Hello World"  # Different case
        text3 = "Hello world!"  # Different punctuation
        model = "test-model"
        
        key1 = cache._make_key(text1, model)
        key2 = cache._make_key(text2, model)
        key3 = cache._make_key(text3, model)
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3