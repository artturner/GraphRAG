"""Tests for the base vector store abstract class.

This module tests the BaseVectorStore abstract class to ensure it cannot be
instantiated directly and that subclasses must implement all required methods.
"""

import pytest

from src.store import BaseVectorStore, SearchResult
from src.types import Chunk, EmbeddingVector


class TestBaseVectorStoreAbstract:
    """Test that BaseVectorStore behaves as an abstract class."""
    
    def test_cannot_instantiate_base_vector_store_directly(self) -> None:
        """Test that BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseVectorStore()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower() or \
               "instantiate" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_add_embeddings_raises_error(self) -> None:
        """Test that a subclass missing add_embeddings cannot be instantiated."""
        # Define incomplete subclass missing add_embeddings
        class IncompleteStoreMissingAddEmbeddings(BaseVectorStore):
            def similarity_search(
                self, query_embedding: EmbeddingVector, k: int
            ) -> list[SearchResult]:
                return []
            
            def delete(self, chunk_ids: list[str]) -> int:
                return 0
            
            def count(self) -> int:
                return 0
            
            def clear(self) -> None:
                pass
            # Missing add_embeddings
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteStoreMissingAddEmbeddings()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_similarity_search_raises_error(self) -> None:
        """Test that a subclass missing similarity_search cannot be instantiated."""
        # Define incomplete subclass missing similarity_search
        class IncompleteStoreMissingSimilaritySearch(BaseVectorStore):
            def add_embeddings(
                self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
            ) -> int:
                return len(chunks)
            
            def delete(self, chunk_ids: list[str]) -> int:
                return 0
            
            def count(self) -> int:
                return 0
            
            def clear(self) -> None:
                pass
            # Missing similarity_search
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteStoreMissingSimilaritySearch()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_delete_raises_error(self) -> None:
        """Test that a subclass missing delete cannot be instantiated."""
        # Define incomplete subclass missing delete
        class IncompleteStoreMissingDelete(BaseVectorStore):
            def add_embeddings(
                self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
            ) -> int:
                return len(chunks)
            
            def similarity_search(
                self, query_embedding: EmbeddingVector, k: int
            ) -> list[SearchResult]:
                return []
            
            def count(self) -> int:
                return 0
            
            def clear(self) -> None:
                pass
            # Missing delete
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteStoreMissingDelete()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_count_raises_error(self) -> None:
        """Test that a subclass missing count cannot be instantiated."""
        # Define incomplete subclass missing count
        class IncompleteStoreMissingCount(BaseVectorStore):
            def add_embeddings(
                self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
            ) -> int:
                return len(chunks)
            
            def similarity_search(
                self, query_embedding: EmbeddingVector, k: int
            ) -> list[SearchResult]:
                return []
            
            def delete(self, chunk_ids: list[str]) -> int:
                return 0
            
            def clear(self) -> None:
                pass
            # Missing count
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteStoreMissingCount()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_incomplete_subclass_missing_clear_raises_error(self) -> None:
        """Test that a subclass missing clear cannot be instantiated."""
        # Define incomplete subclass missing clear
        class IncompleteStoreMissingClear(BaseVectorStore):
            def add_embeddings(
                self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
            ) -> int:
                return len(chunks)
            
            def similarity_search(
                self, query_embedding: EmbeddingVector, k: int
            ) -> list[SearchResult]:
                return []
            
            def delete(self, chunk_ids: list[str]) -> int:
                return 0
            
            def count(self) -> int:
                return 0
            # Missing clear
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteStoreMissingClear()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()


class TestCompleteSubclass:
    """Test that a complete subclass can be instantiated and used correctly."""
    
    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that a complete subclass can be instantiated."""
        class CompleteStore(BaseVectorStore):
            def __init__(self) -> None:
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
        
        store = CompleteStore()
        assert store.count() == 0
    
    def test_complete_subclass_add_embeddings_works(self) -> None:
        """Test that add_embeddings works correctly in a complete subclass."""
        class CompleteStore(BaseVectorStore):
            def __init__(self) -> None:
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
        
        store = CompleteStore()
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        embedding: EmbeddingVector = [0.1, 0.2, 0.3]
        
        result = store.add_embeddings([chunk], [embedding])
        
        assert result == 1
        assert store.count() == 1
    
    def test_complete_subclass_delete_works(self) -> None:
        """Test that delete works correctly in a complete subclass."""
        class CompleteStore(BaseVectorStore):
            def __init__(self) -> None:
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
        
        store = CompleteStore()
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        embedding: EmbeddingVector = [0.1, 0.2, 0.3]
        
        store.add_embeddings([chunk], [embedding])
        deleted = store.delete(["chunk-001"])
        
        assert deleted == 1
        assert store.count() == 0
    
    def test_complete_subclass_clear_works(self) -> None:
        """Test that clear works correctly in a complete subclass."""
        class CompleteStore(BaseVectorStore):
            def __init__(self) -> None:
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
        
        store = CompleteStore()
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        embedding: EmbeddingVector = [0.1, 0.2, 0.3]
        
        store.add_embeddings([chunk], [embedding])
        store.clear()
        
        assert store.count() == 0


class TestSearchResult:
    """Test the SearchResult model."""
    
    def test_search_result_creation(self) -> None:
        """Test that SearchResult can be created with valid data."""
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        
        result = SearchResult(
            chunk_id="chunk-001",
            score=0.95,
            chunk=chunk,
            metadata={"source": "test.txt"}
        )
        
        assert result.chunk_id == "chunk-001"
        assert result.score == 0.95
        assert result.chunk == chunk
        assert result.metadata == {"source": "test.txt"}
    
    def test_search_result_default_metadata(self) -> None:
        """Test that SearchResult has default empty metadata."""
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        
        result = SearchResult(
            chunk_id="chunk-001",
            score=0.95,
            chunk=chunk
        )
        
        assert result.metadata == {}
    
    def test_search_result_frozen(self) -> None:
        """Test that SearchResult is immutable (frozen)."""
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        
        result = SearchResult(
            chunk_id="chunk-001",
            score=0.95,
            chunk=chunk
        )
        
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            result.score = 0.5  # type: ignore[misc]
    
    def test_search_result_score_validation(self) -> None:
        """Test that SearchResult validates score is between 0 and 1."""
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        
        # Valid scores
        result = SearchResult(chunk_id="chunk-001", score=0.0, chunk=chunk)
        assert result.score == 0.0
        
        result = SearchResult(chunk_id="chunk-001", score=1.0, chunk=chunk)
        assert result.score == 1.0
        
        # Invalid scores
        with pytest.raises(Exception):  # Pydantic ValidationError
            SearchResult(chunk_id="chunk-001", score=-0.1, chunk=chunk)
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            SearchResult(chunk_id="chunk-001", score=1.1, chunk=chunk)


class TestExampleFromTask:
    """Test the example usage from the task description."""
    
    def test_example_store_class(self) -> None:
        """Test the example class from the task description works correctly."""
        # This is the exact example from the task
        class MyStore(BaseVectorStore):
            def __init__(self) -> None:
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
        
        store = MyStore()
        
        # Test add_embeddings
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Test content",
            start_idx=0,
            end_idx=12
        )
        embedding: EmbeddingVector = [0.1, 0.2, 0.3]
        
        count = store.add_embeddings([chunk], [embedding])
        assert count == 1
        
        # Test count
        assert store.count() == 1
        
        # Test similarity_search
        results = store.similarity_search([0.1, 0.2, 0.3], k=5)
        assert results == []
        
        # Test delete
        deleted = store.delete(["chunk-001"])
        assert deleted == 1
        
        # Test clear
        store.add_embeddings([chunk], [embedding])
        store.clear()
        assert store.count() == 0