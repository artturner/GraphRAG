"""Tests for the RetrievalService class.

This module tests the RetrievalService to ensure it correctly combines
embeddings and vector store for document indexing and retrieval.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.exceptions import RetrievalError
from src.retrieval import RetrievalService
from src.store.base import SearchResult
from src.types import Chunk


def create_chunk(chunk_id: str, content: str, doc_id: str = "doc-1") -> Chunk:
    """Helper function to create a Chunk object."""
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={"source": "test"},
    )


def create_embedding(dimension: int, seed: int = 42) -> list[float]:
    """Helper function to create a deterministic embedding vector."""
    np.random.seed(seed)
    vector = np.random.randn(dimension).astype(np.float32)
    # Normalize to unit length
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def create_search_result(
    chunk_id: str,
    score: float,
    content: str = "test content",
) -> SearchResult:
    """Helper function to create a SearchResult object."""
    chunk = create_chunk(chunk_id, content)
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=chunk,
        metadata={"source": "test"},
    )


class MockEmbeddings:
    """Mock embeddings provider for testing."""
    
    def __init__(self, dimension: int = 64):
        self._dimension = dimension
        self._call_count = 0
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        self._call_count += 1
        return [create_embedding(self._dimension, seed=i) for i in range(len(texts))]
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query."""
        self._call_count += 1
        return create_embedding(self._dimension, seed=999)


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self, dimension: int = 64):
        self._dimension = dimension
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
    
    def add_embeddings(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add chunks with embeddings to the store."""
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)
        return len(chunks)
    
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        results = []
        for i, chunk in enumerate(self._chunks[:k]):
            results.append(create_search_result(
                chunk_id=chunk.id,
                score=1.0 - (i * 0.1),  # Decreasing scores
                content=chunk.content,
            ))
        return results
    
    def count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)


class TestRetrievalServiceInit:
    """Test RetrievalService initialization."""
    
    def test_init_with_embeddings_and_store(self) -> None:
        """Test initialization with embeddings and store."""
        embeddings = MockEmbeddings(dimension=128)
        store = MockVectorStore(dimension=128)
        
        service = RetrievalService(embeddings, store)
        
        assert service.embeddings is embeddings
        assert service.store is store
    
    def test_init_properties_are_readonly(self) -> None:
        """Test that embeddings and store properties are read-only."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        # Verify we can access but not modify
        assert service.embeddings.dimension == 64
        assert service.store.count() == 0


class TestRetrievalServiceIndexDocuments:
    """Test document indexing functionality."""
    
    def test_index_single_document(self) -> None:
        """Test indexing a single document chunk."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        chunk = create_chunk("chunk-1", "Hello world")
        count = service.index_documents([chunk])
        
        assert count == 1
        assert store.count() == 1
    
    def test_index_multiple_documents(self) -> None:
        """Test indexing multiple document chunks."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        chunks = [
            create_chunk("chunk-1", "First document"),
            create_chunk("chunk-2", "Second document"),
            create_chunk("chunk-3", "Third document"),
        ]
        count = service.index_documents(chunks)
        
        assert count == 3
        assert store.count() == 3
    
    def test_index_empty_chunks_raises_error(self) -> None:
        """Test that indexing empty list raises ValueError."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        with pytest.raises(ValueError, match="Cannot index empty list"):
            service.index_documents([])
    
    def test_index_generates_embeddings(self) -> None:
        """Test that indexing generates embeddings via the provider."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        chunks = [
            create_chunk("chunk-1", "First"),
            create_chunk("chunk-2", "Second"),
        ]
        service.index_documents(chunks)
        
        # embed_documents should have been called once
        assert embeddings._call_count == 1
    
    def test_index_embedding_failure_raises_retrieval_error(self) -> None:
        """Test that embedding failure raises RetrievalError."""
        embeddings = MagicMock()
        embeddings.embed_documents.side_effect = RuntimeError("Embedding failed")
        
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        chunk = create_chunk("chunk-1", "Test")
        
        with pytest.raises(RetrievalError, match="Failed to index documents"):
            service.index_documents([chunk])


class TestRetrievalServiceSearch:
    """Test search functionality."""
    
    def test_search_returns_results(self) -> None:
        """Test that search returns results from the store."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        # Index some documents first
        chunks = [
            create_chunk("chunk-1", "Federalism is a system of government"),
            create_chunk("chunk-2", "The constitution divides power"),
        ]
        service.index_documents(chunks)
        
        results = service.search("What is federalism?")
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_with_k_parameter(self) -> None:
        """Test search with custom k parameter."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        # Index 5 documents
        chunks = [create_chunk(f"chunk-{i}", f"Document {i}") for i in range(5)]
        service.index_documents(chunks)
        
        results = service.search("query", k=3)
        
        assert len(results) == 3
    
    def test_search_empty_query_raises_error(self) -> None:
        """Test that empty query raises ValueError."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.search("")
    
    def test_search_whitespace_query_raises_error(self) -> None:
        """Test that whitespace-only query raises ValueError."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.search("   ")
    
    def test_search_invalid_k_raises_error(self) -> None:
        """Test that invalid k raises ValueError."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        with pytest.raises(ValueError, match="k must be a positive integer"):
            service.search("query", k=0)
        
        with pytest.raises(ValueError, match="k must be a positive integer"):
            service.search("query", k=-1)
    
    def test_search_generates_query_embedding(self) -> None:
        """Test that search generates embedding for the query."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        # Index a document first
        chunk = create_chunk("chunk-1", "Test document")
        service.index_documents([chunk])
        
        # Reset call count after indexing
        embeddings._call_count = 0
        
        service.search("query")
        
        # embed_query should have been called once
        assert embeddings._call_count == 1
    
    def test_search_failure_raises_retrieval_error(self) -> None:
        """Test that search failure raises RetrievalError."""
        embeddings = MagicMock()
        embeddings.embed_query.side_effect = RuntimeError("Search failed")
        
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        with pytest.raises(RetrievalError, match="Search operation failed"):
            service.search("query")


class TestRetrievalServiceSearchWithThreshold:
    """Test search with threshold functionality."""
    
    def test_threshold_filters_low_scores(self) -> None:
        """Test that threshold filters out low-scoring results."""
        embeddings = MockEmbeddings()
        
        # Create a mock store that returns results with varying scores
        store = MagicMock()
        store.similarity_search.return_value = [
            create_search_result("chunk-1", 0.95),
            create_search_result("chunk-2", 0.85),
            create_search_result("chunk-3", 0.75),
            create_search_result("chunk-4", 0.65),
            create_search_result("chunk-5", 0.55),
        ]
        
        service = RetrievalService(embeddings, store)
        
        # Filter with threshold of 0.7
        results = service.search_with_threshold("query", k=5, min_score=0.7)
        
        assert len(results) == 3
        assert all(r.score >= 0.7 for r in results)
    
    def test_threshold_keeps_all_high_scores(self) -> None:
        """Test that threshold keeps all high-scoring results."""
        embeddings = MockEmbeddings()
        
        store = MagicMock()
        store.similarity_search.return_value = [
            create_search_result("chunk-1", 0.99),
            create_search_result("chunk-2", 0.98),
            create_search_result("chunk-3", 0.97),
        ]
        
        service = RetrievalService(embeddings, store)
        
        results = service.search_with_threshold("query", k=5, min_score=0.5)
        
        assert len(results) == 3
    
    def test_threshold_removes_all_low_scores(self) -> None:
        """Test that threshold can remove all results if scores are too low."""
        embeddings = MockEmbeddings()
        
        store = MagicMock()
        store.similarity_search.return_value = [
            create_search_result("chunk-1", 0.3),
            create_search_result("chunk-2", 0.2),
        ]
        
        service = RetrievalService(embeddings, store)
        
        results = service.search_with_threshold("query", k=5, min_score=0.8)
        
        assert len(results) == 0
    
    def test_threshold_invalid_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        embeddings = MockEmbeddings()
        store = MockVectorStore()
        service = RetrievalService(embeddings, store)
        
        with pytest.raises(ValueError, match="min_score must be between 0.0 and 1.0"):
            service.search_with_threshold("query", k=5, min_score=1.5)
        
        with pytest.raises(ValueError, match="min_score must be between 0.0 and 1.0"):
            service.search_with_threshold("query", k=5, min_score=-0.1)
    
    def test_threshold_boundary_values(self) -> None:
        """Test threshold boundary values (0.0 and 1.0)."""
        embeddings = MockEmbeddings()
        
        store = MagicMock()
        store.similarity_search.return_value = [
            create_search_result("chunk-1", 0.5),
        ]
        
        service = RetrievalService(embeddings, store)
        
        # min_score=0.0 should keep all results
        results = service.search_with_threshold("query", k=5, min_score=0.0)
        assert len(results) == 1
        
        # min_score=1.0 should only keep perfect matches
        results = service.search_with_threshold("query", k=5, min_score=1.0)
        assert len(results) == 0


class TestRetrievalServiceIntegration:
    """Integration tests with real embeddings (slow tests)."""
    
    @pytest.mark.slow
    def test_with_real_embeddings(self) -> None:
        """Test with real LocalEmbeddings (slow test).
        
        This test verifies the full pipeline works with actual embeddings.
        """
        # Import here to avoid slow imports during other tests
        from src.embeddings.local import LocalEmbeddings
        from src.store.faiss_store import FAISSVectorStore
        
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        store = FAISSVectorStore(dimension=embeddings.dimension)
        service = RetrievalService(embeddings, store)
        
        # Index some documents
        chunks = [
            create_chunk(
                "chunk-1",
                "Federalism is a system of government where power is divided "
                "between a central authority and constituent political units.",
            ),
            create_chunk(
                "chunk-2",
                "The United States Constitution establishes a federal system "
                "with powers divided between federal and state governments.",
            ),
            create_chunk(
                "chunk-3",
                "Machine learning is a subset of artificial intelligence that "
                "enables systems to learn and improve from experience.",
            ),
        ]
        
        count = service.index_documents(chunks)
        assert count == 3
        
        # Search for relevant documents
        results = service.search("What is federalism?", k=2)
        
        assert len(results) == 2
        # The first result should be about federalism
        assert "federalism" in results[0].chunk.content.lower()
        assert results[0].score > results[1].score
        
        # Test with threshold
        high_score_results = service.search_with_threshold(
            "What is federalism?",
            k=3,
            min_score=0.5,
        )
        
        # All results should meet the threshold
        assert all(r.score >= 0.5 for r in high_score_results)
    
    @pytest.mark.slow
    def test_search_returns_empty_for_unrelated_content(self) -> None:
        """Test that search returns low scores for unrelated content."""
        from src.embeddings.local import LocalEmbeddings
        from src.store.faiss_store import FAISSVectorStore
        
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        store = FAISSVectorStore(dimension=embeddings.dimension)
        service = RetrievalService(embeddings, store)
        
        # Index documents about cooking
        chunks = [
            create_chunk(
                "chunk-1",
                "To make pasta, boil water and add salt before cooking.",
            ),
            create_chunk(
                "chunk-2",
                "Baking requires precise measurements of ingredients.",
            ),
        ]
        
        service.index_documents(chunks)
        
        # Search for something unrelated
        results = service.search("quantum physics and particle accelerators", k=2)
        
        # Results should have low scores for unrelated content
        assert all(r.score < 0.5 for r in results)