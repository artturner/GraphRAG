"""Tests for the base embeddings abstract class.

This module tests the BaseEmbeddings abstract class to ensure it cannot be
instantiated directly and that subclasses must implement all required methods.
"""

import pytest

from src.embeddings import BaseEmbeddings, EmbeddingCache
from src.types import EmbeddingVector


class TestBaseEmbeddingsAbstract:
    """Test that BaseEmbeddings behaves as an abstract class."""
    
    def test_cannot_instantiate_base_embeddings_directly(self) -> None:
        """Test that BaseEmbeddings cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseEmbeddings()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower() or \
               "instantiate" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_raises_error(self) -> None:
        """Test that a subclass missing methods cannot be instantiated."""
        # Define incomplete subclass missing embed_query
        class IncompleteEmbeddingsMissingEmbedQuery(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            # Missing embed_query
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteEmbeddingsMissingEmbedQuery()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_dimension_raises_error(self) -> None:
        """Test that a subclass missing dimension property cannot be instantiated."""
        # Define incomplete subclass missing dimension
        class IncompleteEmbeddingsMissingDimension(BaseEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
            # Missing dimension property
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteEmbeddingsMissingDimension()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_embed_documents_raises_error(self) -> None:
        """Test that a subclass missing embed_documents cannot be instantiated."""
        # Define incomplete subclass missing embed_documents
        class IncompleteEmbeddingsMissingEmbedDocuments(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
            # Missing embed_documents
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteEmbeddingsMissingEmbedDocuments()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()


class TestCompleteSubclass:
    """Test that a complete subclass can be instantiated and used correctly."""
    
    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that a complete subclass can be instantiated."""
        class CompleteEmbeddings(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
        
        embeddings = CompleteEmbeddings()
        assert embeddings.dimension == 768
    
    def test_complete_subclass_embed_documents_works(self) -> None:
        """Test that embed_documents works correctly in a complete subclass."""
        class CompleteEmbeddings(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
        
        embeddings = CompleteEmbeddings()
        texts = ["Hello world", "Goodbye world"]
        vectors = embeddings.embed_documents(texts)
        
        assert len(vectors) == 2
        assert all(len(v) == 768 for v in vectors)
        assert all(all(x == 0.1 for x in v) for v in vectors)
    
    def test_complete_subclass_embed_query_works(self) -> None:
        """Test that embed_query works correctly in a complete subclass."""
        class CompleteEmbeddings(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
        
        embeddings = CompleteEmbeddings()
        vector = embeddings.embed_query("What is machine learning?")
        
        assert len(vector) == 768
        assert all(x == 0.1 for x in vector)
    
    def test_dimension_property_is_required(self) -> None:
        """Test that the dimension property is required and returns int."""
        class CompleteEmbeddings(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 1536  # Different dimension
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.0] * 1536 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.0] * 1536
        
        embeddings = CompleteEmbeddings()
        assert embeddings.dimension == 1536
        assert isinstance(embeddings.dimension, int)


class TestEmbeddingCacheProtocol:
    """Test the EmbeddingCache protocol."""
    
    def test_embedding_cache_protocol_is_runtime_checkable(self) -> None:
        """Test that EmbeddingCache is runtime checkable."""
        class InMemoryCache:
            def __init__(self) -> None:
                self._cache: dict[str, EmbeddingVector] = {}
            
            def get(self, key: str) -> EmbeddingVector | None:
                return self._cache.get(key)
            
            def set(self, key: str, value: EmbeddingVector) -> None:
                self._cache[key] = value
            
            def contains(self, key: str) -> bool:
                return key in self._cache
            
            def clear(self) -> None:
                self._cache.clear()
        
        cache = InMemoryCache()
        assert isinstance(cache, EmbeddingCache)
    
    def test_embedding_cache_protocol_methods(self) -> None:
        """Test that a cache implementation works correctly."""
        class InMemoryCache:
            def __init__(self) -> None:
                self._cache: dict[str, EmbeddingVector] = {}
            
            def get(self, key: str) -> EmbeddingVector | None:
                return self._cache.get(key)
            
            def set(self, key: str, value: EmbeddingVector) -> None:
                self._cache[key] = value
            
            def contains(self, key: str) -> bool:
                return key in self._cache
            
            def clear(self) -> None:
                self._cache.clear()
        
        cache = InMemoryCache()
        
        # Test contains returns False for missing key
        assert not cache.contains("test_key")
        
        # Test get returns None for missing key
        assert cache.get("test_key") is None
        
        # Test set and get
        test_vector: EmbeddingVector = [0.1, 0.2, 0.3]
        cache.set("test_key", test_vector)
        assert cache.contains("test_key")
        assert cache.get("test_key") == test_vector
        
        # Test clear
        cache.clear()
        assert not cache.contains("test_key")
        assert cache.get("test_key") is None
    
    def test_incomplete_cache_not_recognized_as_protocol(self) -> None:
        """Test that an incomplete cache implementation is not recognized."""
        class IncompleteCache:
            def get(self, key: str) -> EmbeddingVector | None:
                return None
            # Missing set, contains, clear
        
        cache = IncompleteCache()
        assert not isinstance(cache, EmbeddingCache)


class TestExampleFromTask:
    """Test the example usage from the task description."""
    
    def test_example_embeddings_class(self) -> None:
        """Test the example class from the task description works correctly."""
        # This is the exact example from the task
        class MyEmbeddings(BaseEmbeddings):
            @property
            def dimension(self) -> int:
                return 768
            
            def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text: str) -> EmbeddingVector:
                return [0.1] * 768
        
        embeddings = MyEmbeddings()
        
        # Test dimension
        assert embeddings.dimension == 768
        
        # Test embed_documents
        texts = ["Hello", "World"]
        doc_vectors = embeddings.embed_documents(texts)
        assert len(doc_vectors) == 2
        assert all(len(v) == 768 for v in doc_vectors)
        
        # Test embed_query
        query_vector = embeddings.embed_query("Test query")
        assert len(query_vector) == 768