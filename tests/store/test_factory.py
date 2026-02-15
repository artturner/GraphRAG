"""Tests for the VectorStoreFactory class.

This module tests the factory functionality for creating vector store
instances based on configuration.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.config import VectorStoreConfig
from src.exceptions import ConfigurationError
from src.store import VectorStoreFactory
from src.store.base import BaseVectorStore
from src.store.faiss_store import FAISSVectorStore
from src.store.chroma_store import ChromaVectorStore


class TestVectorStoreFactory:
    """Test cases for VectorStoreFactory class."""
    
    def test_get_store_returns_faiss_for_faiss_type(self) -> None:
        """Test that factory returns FAISSVectorStore for 'faiss' type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(type="faiss", persist_directory=tmpdir)
            store = VectorStoreFactory.get_store(config, dimension=384)
            
            assert isinstance(store, FAISSVectorStore)
            assert isinstance(store, BaseVectorStore)
    
    def test_get_store_returns_chroma_for_chroma_type(self) -> None:
        """Test that factory returns ChromaVectorStore for 'chroma' type."""
        tmpdir = tempfile.mkdtemp()
        try:
            config = VectorStoreConfig(
                type="chroma",
                persist_directory=tmpdir,
                collection_name="test_collection"
            )
            store = VectorStoreFactory.get_store(config, dimension=384)
            
            assert isinstance(store, ChromaVectorStore)
            assert isinstance(store, BaseVectorStore)
        finally:
            # ChromaDB holds file locks, so we just leave the temp dir
            pass
    
    def test_get_store_raises_error_for_unsupported_type(self) -> None:
        """Test that factory raises ConfigurationError for unsupported type."""
        # Create config with invalid type by bypassing validation
        config = VectorStoreConfig(type="faiss")  # Valid type first
        # Manually set to invalid type to bypass validation
        object.__setattr__(config, "type", "unsupported")
        
        with pytest.raises(ConfigurationError) as exc_info:
            VectorStoreFactory.get_store(config, dimension=384)
        
        assert "Unsupported vector store type" in str(exc_info.value)
        assert "unsupported" in str(exc_info.value)
    
    def test_dimension_passed_correctly_to_faiss(self) -> None:
        """Test that dimension is correctly passed to FAISS store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(type="faiss", persist_directory=tmpdir)
            
            store = VectorStoreFactory.get_store(config, dimension=128)
            assert store._dimension == 128
            
            store = VectorStoreFactory.get_store(config, dimension=768)
            assert store._dimension == 768
    
    def test_dimension_passed_correctly_to_faiss_high_dimension(self) -> None:
        """Test that high dimensions are correctly passed to FAISS store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(type="faiss", persist_directory=tmpdir)
            
            store = VectorStoreFactory.get_store(config, dimension=1536)
            assert store._dimension == 1536
    
    def test_persist_directory_passed_to_faiss(self) -> None:
        """Test that persist_directory is passed to FAISS store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(
                type="faiss",
                persist_directory=tmpdir
            )
            store = VectorStoreFactory.get_store(config, dimension=384)
            
            assert store._persist_dir is not None
            assert tmpdir in str(store._persist_dir)
    
    def test_collection_name_passed_to_chroma(self) -> None:
        """Test that collection_name is passed to ChromaDB store."""
        tmpdir = tempfile.mkdtemp()
        try:
            config = VectorStoreConfig(
                type="chroma",
                persist_directory=tmpdir,
                collection_name="my_collection"
            )
            store = VectorStoreFactory.get_store(config, dimension=384)
            
            assert store._collection_name == "my_collection"
        finally:
            pass
    
    def test_persist_directory_passed_to_chroma(self) -> None:
        """Test that persist_directory is passed to ChromaDB store."""
        tmpdir = tempfile.mkdtemp()
        try:
            config = VectorStoreConfig(
                type="chroma",
                persist_directory=tmpdir,
                collection_name="test_collection"
            )
            store = VectorStoreFactory.get_store(config, dimension=384)
            
            assert store._persist_dir is not None
            assert tmpdir in str(store._persist_dir)
        finally:
            pass
    
    def test_get_supported_types_returns_sorted_list(self) -> None:
        """Test that get_supported_types returns sorted list of types."""
        types = VectorStoreFactory.get_supported_types()
        
        assert types == ["chroma", "faiss"]
        assert isinstance(types, list)
    
    def test_is_type_supported_returns_true_for_valid_types(self) -> None:
        """Test that is_type_supported returns True for valid types."""
        assert VectorStoreFactory.is_type_supported("faiss") is True
        assert VectorStoreFactory.is_type_supported("chroma") is True
    
    def test_is_type_supported_returns_false_for_invalid_type(self) -> None:
        """Test that is_type_supported returns False for invalid type."""
        assert VectorStoreFactory.is_type_supported("invalid") is False
        assert VectorStoreFactory.is_type_supported("pinecone") is False
    
    def test_register_store_adds_new_type(self) -> None:
        """Test that register_store adds a new store type."""
        # Create a mock store class
        class MockStore(BaseVectorStore):
            def __init__(self, dimension: int = 384):
                self._dimension = dimension
            
            def add_embeddings(self, chunks, embeddings):
                return 0
            
            def similarity_search(self, query_embedding, k):
                return []
            
            def delete(self, chunk_ids):
                return 0
            
            def count(self):
                return 0
            
            def clear(self):
                pass
        
        # Register the mock store
        VectorStoreFactory.register_store("mock", MockStore)
        
        try:
            assert VectorStoreFactory.is_type_supported("mock") is True
            assert "mock" in VectorStoreFactory.get_supported_types()
        finally:
            # Clean up
            VectorStoreFactory.reset()
    
    def test_register_store_raises_error_for_invalid_class(self) -> None:
        """Test that register_store raises TypeError for invalid class."""
        with pytest.raises(TypeError) as exc_info:
            VectorStoreFactory.register_store("invalid", str)  # str is not a BaseVectorStore
        
        assert "must be a subclass of BaseVectorStore" in str(exc_info.value)
    
    def test_register_store_raises_error_for_non_class(self) -> None:
        """Test that register_store raises TypeError for non-class."""
        with pytest.raises(TypeError) as exc_info:
            VectorStoreFactory.register_store("invalid", "not a class")
        
        assert "must be a subclass of BaseVectorStore" in str(exc_info.value)
    
    def test_clear_removes_all_stores(self) -> None:
        """Test that clear removes all registered stores."""
        original_types = VectorStoreFactory.get_supported_types()
        
        VectorStoreFactory.clear()
        
        try:
            assert VectorStoreFactory.get_supported_types() == []
            assert VectorStoreFactory.is_type_supported("faiss") is False
        finally:
            # Restore original state
            VectorStoreFactory.reset()
        
        # Verify restoration
        assert VectorStoreFactory.get_supported_types() == original_types
    
    def test_reset_restores_default_stores(self) -> None:
        """Test that reset restores default store registrations."""
        # Clear and modify
        VectorStoreFactory.clear()
        
        # Reset
        VectorStoreFactory.reset()
        
        # Verify defaults are restored
        types = VectorStoreFactory.get_supported_types()
        assert types == ["chroma", "faiss"]
        assert VectorStoreFactory.is_type_supported("faiss") is True
        assert VectorStoreFactory.is_type_supported("chroma") is True
    
    def test_factory_with_different_dimensions(self) -> None:
        """Test factory with various dimension values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(type="faiss", persist_directory=tmpdir)
            
            dimensions = [64, 128, 256, 384, 512, 768, 1024, 1536]
            
            for dim in dimensions:
                store = VectorStoreFactory.get_store(config, dimension=dim)
                assert store._dimension == dim


class TestVectorStoreFactoryIntegration:
    """Integration tests for VectorStoreFactory."""
    
    def test_faiss_store_functional(self) -> None:
        """Test that created FAISS store is functional."""
        from src.types import Chunk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(type="faiss", persist_directory=tmpdir)
            store = VectorStoreFactory.get_store(config, dimension=4)
            
            # Create test data
            chunks = [
                Chunk(id="1", document_id="doc1", content="test 1", start_idx=0, end_idx=6, metadata={}),
                Chunk(id="2", document_id="doc1", content="test 2", start_idx=7, end_idx=13, metadata={}),
            ]
            embeddings = [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
            
            # Test add_embeddings
            count = store.add_embeddings(chunks, embeddings)
            assert count == 2
            assert store.count() == 2
            
            # Test similarity_search
            results = store.similarity_search([0.1, 0.2, 0.3, 0.4], k=1)
            assert len(results) == 1
            assert results[0].chunk_id == "1"
    
    def test_chroma_store_functional(self) -> None:
        """Test that created ChromaDB store is functional."""
        from src.types import Chunk
        
        tmpdir = tempfile.mkdtemp()
        try:
            config = VectorStoreConfig(
                type="chroma",
                persist_directory=tmpdir,
                collection_name="test_integration"
            )
            store = VectorStoreFactory.get_store(config, dimension=4)
            
            # Create test data
            chunks = [
                Chunk(id="1", document_id="doc1", content="test 1", start_idx=0, end_idx=6, metadata={}),
                Chunk(id="2", document_id="doc1", content="test 2", start_idx=7, end_idx=13, metadata={}),
            ]
            embeddings = [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
            
            # Test add_embeddings
            count = store.add_embeddings(chunks, embeddings)
            assert count == 2
            assert store.count() == 2
            
            # Test similarity_search
            results = store.similarity_search([0.1, 0.2, 0.3, 0.4], k=1)
            assert len(results) == 1
            
            # Clean up
            store.clear()
        finally:
            pass


class TestVectorStoreFactoryEdgeCases:
    """Edge case tests for VectorStoreFactory."""
    
    def test_faiss_with_default_persist_directory(self) -> None:
        """Test FAISS store with default persist directory."""
        config = VectorStoreConfig(type="faiss")  # Use default
        store = VectorStoreFactory.get_store(config, dimension=384)
        
        # Default is ./.vectorstore
        assert store._persist_dir is not None
    
    def test_chroma_with_default_persist_directory(self) -> None:
        """Test ChromaDB store with default persist directory."""
        config = VectorStoreConfig(type="chroma", collection_name="default_test")
        store = VectorStoreFactory.get_store(config, dimension=384)
        
        # Default is ./.vectorstore
        assert store._persist_dir is not None
    
    def test_multiple_faiss_stores_independent(self) -> None:
        """Test that multiple FAISS stores are independent."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                config1 = VectorStoreConfig(type="faiss", persist_directory=tmpdir1)
                config2 = VectorStoreConfig(type="faiss", persist_directory=tmpdir2)
                
                store1 = VectorStoreFactory.get_store(config1, dimension=4)
                store2 = VectorStoreFactory.get_store(config2, dimension=4)
                
                # Add data to store1
                from src.types import Chunk
                chunks = [Chunk(id="1", document_id="doc1", content="test", start_idx=0, end_idx=4, metadata={})]
                embeddings = [[0.1, 0.2, 0.3, 0.4]]
                store1.add_embeddings(chunks, embeddings)
                
                # store2 should be independent
                assert store1.count() == 1
                assert store2.count() == 0
    
    def test_multiple_chroma_stores_different_collections(self) -> None:
        """Test that multiple ChromaDB stores with different collections are independent."""
        tmpdir = tempfile.mkdtemp()
        try:
            config1 = VectorStoreConfig(
                type="chroma",
                persist_directory=tmpdir,
                collection_name="collection_1"
            )
            config2 = VectorStoreConfig(
                type="chroma",
                persist_directory=tmpdir,
                collection_name="collection_2"
            )
            
            store1 = VectorStoreFactory.get_store(config1, dimension=4)
            store2 = VectorStoreFactory.get_store(config2, dimension=4)
            
            assert store1._collection_name == "collection_1"
            assert store2._collection_name == "collection_2"
            
            # Clean up
            store1.clear()
            store2.clear()
        finally:
            pass