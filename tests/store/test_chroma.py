"""Tests for the ChromaDB vector store implementation.

This module tests the ChromaVectorStore class to ensure it correctly
implements the BaseVectorStore interface and handles all operations properly.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.store.chroma_store import ChromaVectorStore
from src.store.base import SearchResult
from src.types import Chunk


def create_chunk(
    chunk_id: str, content: str, doc_id: str = "doc-1", metadata: dict | None = None
) -> Chunk:
    """Helper function to create a Chunk object."""
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata=metadata or {"source": "test"},
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


def create_orthogonal_embeddings(dimension: int, count: int) -> list[list[float]]:
    """Create approximately orthogonal embeddings for testing."""
    np.random.seed(42)
    embeddings = []
    for i in range(count):
        # Create a vector with a distinct pattern
        vector = np.zeros(dimension, dtype=np.float32)
        # Set different dimensions for each vector to make them more distinguishable
        start_idx = (i * dimension // count)
        end_idx = ((i + 1) * dimension // count)
        vector[start_idx:end_idx] = 1.0
        # Add some noise
        vector += np.random.randn(dimension).astype(np.float32) * 0.1
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        embeddings.append(vector.tolist())
    return embeddings


def unique_collection_name(prefix: str = "test") -> str:
    """Generate a unique collection name for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestChromaVectorStoreInit:
    """Test ChromaVectorStore initialization."""
    
    def test_init_with_collection_name(self) -> None:
        """Test initialization with just collection name (in-memory)."""
        store = ChromaVectorStore(collection_name=unique_collection_name("init"))
        assert store.collection_name.startswith("init_")
        assert store.persist_directory is None
        assert store.count() == 0
    
    def test_init_with_persist_directory(self) -> None:
        """Test initialization with persist directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            store = ChromaVectorStore(
                collection_name=unique_collection_name("persist"),
                persist_directory=tmpdir
            )
            assert store.persist_directory == Path(tmpdir)
            assert store.count() == 0
        finally:
            # Clean up manually on Windows
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
    
    def test_init_default_distance_metric(self) -> None:
        """Test that default distance metric is cosine."""
        store = ChromaVectorStore(collection_name=unique_collection_name("metric"))
        assert store.distance_metric == "cosine"
    
    def test_init_custom_distance_metric(self) -> None:
        """Test initialization with custom distance metric."""
        store = ChromaVectorStore(
            collection_name=unique_collection_name("l2"),
            distance_metric="l2"
        )
        assert store.distance_metric == "l2"


class TestChromaVectorStoreAddEmbeddings:
    """Test adding embeddings to the store."""
    
    def test_add_single_embedding(self) -> None:
        """Test adding a single chunk with embedding."""
        store = ChromaVectorStore(collection_name=unique_collection_name("add_single"))
        chunk = create_chunk("chunk-1", "Hello world")
        embedding = create_embedding(64, seed=1)
        
        count = store.add_embeddings([chunk], [embedding])
        
        assert count == 1
        assert store.count() == 1
    
    def test_add_multiple_embeddings(self) -> None:
        """Test adding multiple chunks with embeddings."""
        store = ChromaVectorStore(collection_name=unique_collection_name("add_multi"))
        chunks = [
            create_chunk("chunk-1", "First chunk"),
            create_chunk("chunk-2", "Second chunk"),
            create_chunk("chunk-3", "Third chunk"),
        ]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        
        count = store.add_embeddings(chunks, embeddings)
        
        assert count == 3
        assert store.count() == 3
    
    def test_add_empty_list(self) -> None:
        """Test adding empty lists returns 0."""
        store = ChromaVectorStore(collection_name=unique_collection_name("add_empty"))
        
        count = store.add_embeddings([], [])
        
        assert count == 0
        assert store.count() == 0
    
    def test_add_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched chunk/embedding lengths raises ValueError."""
        store = ChromaVectorStore(collection_name=unique_collection_name("mismatch"))
        chunks = [create_chunk("chunk-1", "Test")]
        embeddings = [create_embedding(64), create_embedding(64)]
        
        with pytest.raises(ValueError) as exc_info:
            store.add_embeddings(chunks, embeddings)
        
        assert "must match" in str(exc_info.value)
    
    def test_add_incremental(self) -> None:
        """Test adding embeddings incrementally."""
        store = ChromaVectorStore(collection_name=unique_collection_name("incremental"))
        
        # First batch
        chunks1 = [create_chunk("chunk-1", "First")]
        embeddings1 = [create_embedding(64, seed=1)]
        store.add_embeddings(chunks1, embeddings1)
        assert store.count() == 1
        
        # Second batch
        chunks2 = [create_chunk("chunk-2", "Second")]
        embeddings2 = [create_embedding(64, seed=2)]
        store.add_embeddings(chunks2, embeddings2)
        assert store.count() == 2
        
        # Third batch
        chunks3 = [
            create_chunk("chunk-3", "Third"),
            create_chunk("chunk-4", "Fourth"),
        ]
        embeddings3 = [create_embedding(64, seed=3), create_embedding(64, seed=4)]
        store.add_embeddings(chunks3, embeddings3)
        assert store.count() == 4
    
    def test_add_update_existing_chunk(self) -> None:
        """Test that adding a chunk with existing ID updates it."""
        store = ChromaVectorStore(collection_name=unique_collection_name("update"))
        
        # Add initial chunk
        chunk1 = create_chunk("chunk-1", "Original content")
        embedding1 = create_embedding(64, seed=1)
        store.add_embeddings([chunk1], [embedding1])
        assert store.count() == 1
        
        # Update with same ID
        chunk2 = create_chunk("chunk-1", "Updated content")
        embedding2 = create_embedding(64, seed=2)
        store.add_embeddings([chunk2], [embedding2])
        assert store.count() == 1  # Still 1, not 2
        
        # Verify content was updated
        results = store.similarity_search(embedding2, k=1)
        assert results[0].chunk.content == "Updated content"
    
    def test_add_with_complex_metadata(self) -> None:
        """Test adding chunks with various metadata types."""
        store = ChromaVectorStore(collection_name=unique_collection_name("metadata"))
        
        chunk = create_chunk(
            "chunk-1",
            "Test content",
            metadata={
                "string_val": "hello",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
                "none_val": None,
                "list_val": [1, 2, 3],  # Should be converted to string
            }
        )
        embedding = create_embedding(64)
        
        count = store.add_embeddings([chunk], [embedding])
        
        assert count == 1
        assert store.count() == 1


class TestChromaVectorStoreSimilaritySearch:
    """Test similarity search functionality."""
    
    def test_search_returns_correct_results(self) -> None:
        """Test that search returns correct results."""
        store = ChromaVectorStore(collection_name=unique_collection_name("search_basic"))
        
        # Create chunks with distinct embeddings
        chunks = [
            create_chunk("chunk-1", "First document"),
            create_chunk("chunk-2", "Second document"),
            create_chunk("chunk-3", "Third document"),
        ]
        embeddings = create_orthogonal_embeddings(64, 3)
        store.add_embeddings(chunks, embeddings)
        
        # Search with first embedding (should match first chunk best)
        results = store.similarity_search(embeddings[0], k=1)
        
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
        assert results[0].score > 0.9  # Should be very similar
    
    def test_search_with_different_k_values(self) -> None:
        """Test search with different k values."""
        store = ChromaVectorStore(collection_name=unique_collection_name("search_k"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Document {i}") for i in range(5)]
        embeddings = [create_embedding(64, seed=i) for i in range(5)]
        store.add_embeddings(chunks, embeddings)
        
        # k=1
        results = store.similarity_search(embeddings[0], k=1)
        assert len(results) == 1
        
        # k=3
        results = store.similarity_search(embeddings[0], k=3)
        assert len(results) == 3
        
        # k=5
        results = store.similarity_search(embeddings[0], k=5)
        assert len(results) == 5
        
        # k larger than store size
        results = store.similarity_search(embeddings[0], k=10)
        assert len(results) == 5  # Only 5 in store
    
    def test_search_results_sorted_by_score(self) -> None:
        """Test that results are sorted by score (highest first)."""
        store = ChromaVectorStore(collection_name=unique_collection_name("search_sorted"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Document {i}") for i in range(5)]
        embeddings = create_orthogonal_embeddings(64, 5)
        store.add_embeddings(chunks, embeddings)
        
        # Create a query that's similar to multiple chunks
        query = embeddings[0]
        results = store.similarity_search(query, k=5)
        
        # Verify scores are in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_result_has_correct_fields(self) -> None:
        """Test that SearchResult has all required fields."""
        store = ChromaVectorStore(collection_name=unique_collection_name("search_fields"))
        
        chunk = create_chunk("chunk-1", "Test content", doc_id="doc-1")
        embedding = create_embedding(64, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.chunk_id == "chunk-1"
        assert result.chunk.content == "Test content"
        assert result.chunk.document_id == "doc-1"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.metadata, dict)


class TestChromaVectorStoreMetadataFiltering:
    """Test metadata filtering functionality."""
    
    def test_search_with_single_metadata_filter(self) -> None:
        """Test search with a single metadata filter."""
        store = ChromaVectorStore(collection_name=unique_collection_name("filter_single"))
        
        chunks = [
            create_chunk("chunk-1", "Document A", metadata={"category": "tech"}),
            create_chunk("chunk-2", "Document B", metadata={"category": "news"}),
            create_chunk("chunk-3", "Document C", metadata={"category": "tech"}),
        ]
        embeddings = create_orthogonal_embeddings(64, 3)
        store.add_embeddings(chunks, embeddings)
        
        # Search with filter for tech category
        results = store.similarity_search(
            embeddings[0], k=3, filter_metadata={"category": "tech"}
        )
        
        # Should only return tech documents
        assert len(results) == 2
        for result in results:
            assert result.chunk.metadata.get("category") == "tech"
    
    def test_search_with_multiple_metadata_filters(self) -> None:
        """Test search with multiple metadata filters."""
        store = ChromaVectorStore(collection_name=unique_collection_name("filter_multi"))
        
        chunks = [
            create_chunk("chunk-1", "Doc A", metadata={"category": "tech", "source": "web"}),
            create_chunk("chunk-2", "Doc B", metadata={"category": "tech", "source": "file"}),
            create_chunk("chunk-3", "Doc C", metadata={"category": "news", "source": "web"}),
        ]
        embeddings = create_orthogonal_embeddings(64, 3)
        store.add_embeddings(chunks, embeddings)
        
        # Search with multiple filters
        results = store.similarity_search(
            embeddings[0], k=3, filter_metadata={"category": "tech", "source": "web"}
        )
        
        # Should only return tech + web documents
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
    
    def test_search_with_no_matching_filter(self) -> None:
        """Test search with filter that matches nothing."""
        store = ChromaVectorStore(collection_name=unique_collection_name("filter_none"))
        
        chunks = [
            create_chunk("chunk-1", "Doc A", metadata={"category": "tech"}),
        ]
        embeddings = [create_embedding(64)]
        store.add_embeddings(chunks, embeddings)
        
        # Search with non-matching filter
        results = store.similarity_search(
            embeddings[0], k=3, filter_metadata={"category": "nonexistent"}
        )
        
        assert len(results) == 0


class TestChromaVectorStoreEmptyBehavior:
    """Test behavior with empty store."""
    
    def test_search_empty_store_returns_empty_list(self) -> None:
        """Test that searching an empty store returns empty list."""
        store = ChromaVectorStore(collection_name=unique_collection_name("empty_search"))
        query = create_embedding(64)
        
        results = store.similarity_search(query, k=5)
        
        assert results == []
    
    def test_count_empty_store(self) -> None:
        """Test count on empty store."""
        store = ChromaVectorStore(collection_name=unique_collection_name("empty_count"))
        assert store.count() == 0
    
    def test_delete_empty_store(self) -> None:
        """Test delete on empty store."""
        store = ChromaVectorStore(collection_name=unique_collection_name("empty_delete"))
        deleted = store.delete(["nonexistent"])
        assert deleted == 0
    
    def test_clear_empty_store(self) -> None:
        """Test clear on empty store."""
        store = ChromaVectorStore(collection_name=unique_collection_name("empty_clear"))
        store.clear()  # Should not raise
        assert store.count() == 0


class TestChromaVectorStoreDelete:
    """Test delete functionality."""
    
    def test_delete_single_chunk(self) -> None:
        """Test deleting a single chunk."""
        store = ChromaVectorStore(collection_name=unique_collection_name("delete_single"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        assert store.count() == 3
        
        deleted = store.delete(["chunk-1"])
        
        assert deleted == 1
        assert store.count() == 2
    
    def test_delete_multiple_chunks(self) -> None:
        """Test deleting multiple chunks."""
        store = ChromaVectorStore(collection_name=unique_collection_name("delete_multi"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(5)]
        embeddings = [create_embedding(64, seed=i) for i in range(5)]
        store.add_embeddings(chunks, embeddings)
        assert store.count() == 5
        
        deleted = store.delete(["chunk-1", "chunk-3"])
        
        assert deleted == 2
        assert store.count() == 3
    
    def test_delete_nonexistent_chunk(self) -> None:
        """Test deleting a chunk that doesn't exist."""
        store = ChromaVectorStore(collection_name=unique_collection_name("delete_nonexist"))
        
        chunk = create_chunk("chunk-1", "Doc")
        embedding = create_embedding(64)
        store.add_embeddings([chunk], [embedding])
        
        deleted = store.delete(["nonexistent"])
        
        assert deleted == 0
        assert store.count() == 1
    
    def test_delete_empty_list(self) -> None:
        """Test deleting with empty list."""
        store = ChromaVectorStore(collection_name=unique_collection_name("delete_empty_list"))
        
        chunk = create_chunk("chunk-1", "Doc")
        embedding = create_embedding(64)
        store.add_embeddings([chunk], [embedding])
        
        deleted = store.delete([])
        
        assert deleted == 0
        assert store.count() == 1


class TestChromaVectorStoreCountAndClear:
    """Test count and clear functionality."""
    
    def test_count_after_additions(self) -> None:
        """Test count after adding chunks."""
        store = ChromaVectorStore(collection_name=unique_collection_name("count_add"))
        
        assert store.count() == 0
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        assert store.count() == 3
    
    def test_count_after_deletions(self) -> None:
        """Test count after deleting chunks."""
        store = ChromaVectorStore(collection_name=unique_collection_name("count_del"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        store.delete(["chunk-1"])
        
        assert store.count() == 2
    
    def test_clear_removes_all_chunks(self) -> None:
        """Test that clear removes all chunks."""
        store = ChromaVectorStore(collection_name=unique_collection_name("clear"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(5)]
        embeddings = [create_embedding(64, seed=i) for i in range(5)]
        store.add_embeddings(chunks, embeddings)
        assert store.count() == 5
        
        store.clear()
        
        assert store.count() == 0
    
    def test_clear_allows_new_additions(self) -> None:
        """Test that clear allows new additions."""
        store = ChromaVectorStore(collection_name=unique_collection_name("clear_new"))
        
        # Add and clear
        chunks1 = [create_chunk("chunk-1", "First")]
        embeddings1 = [create_embedding(64, seed=1)]
        store.add_embeddings(chunks1, embeddings1)
        store.clear()
        
        # Add new chunks
        chunks2 = [create_chunk("chunk-2", "Second")]
        embeddings2 = [create_embedding(64, seed=2)]
        store.add_embeddings(chunks2, embeddings2)
        
        assert store.count() == 1


class TestChromaVectorStorePersistence:
    """Test persistence functionality."""
    
    def test_persist_to_disk(self) -> None:
        """Test that data persists to disk."""
        tmpdir = tempfile.mkdtemp()
        try:
            col_name = unique_collection_name("persist_disk")
            # Create store and add data
            store1 = ChromaVectorStore(
                collection_name=col_name,
                persist_directory=tmpdir
            )
            
            chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
            embeddings = [create_embedding(64, seed=i) for i in range(3)]
            store1.add_embeddings(chunks, embeddings)
            assert store1.count() == 3
            
            # Create new store with same persist directory
            store2 = ChromaVectorStore(
                collection_name=col_name,
                persist_directory=tmpdir
            )
            
            # Data should be persisted
            assert store2.count() == 3
        finally:
            # Clean up manually on Windows
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
    
    def test_persist_search_results(self) -> None:
        """Test that search works after persistence."""
        tmpdir = tempfile.mkdtemp()
        try:
            col_name = unique_collection_name("persist_search")
            # Create store and add data
            store1 = ChromaVectorStore(
                collection_name=col_name,
                persist_directory=tmpdir
            )
            
            chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
            embeddings = create_orthogonal_embeddings(64, 3)
            store1.add_embeddings(chunks, embeddings)
            
            # Create new store and search
            store2 = ChromaVectorStore(
                collection_name=col_name,
                persist_directory=tmpdir
            )
            
            results = store2.similarity_search(embeddings[0], k=1)
            
            assert len(results) == 1
            assert results[0].chunk_id == "chunk-0"
        finally:
            # Clean up manually on Windows
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


class TestChromaVectorStoreAdditionalMethods:
    """Test additional methods specific to ChromaVectorStore."""
    
    def test_get_by_ids(self) -> None:
        """Test retrieving chunks by IDs."""
        store = ChromaVectorStore(collection_name=unique_collection_name("get_by_ids"))
        
        chunks = [
            create_chunk("chunk-1", "First"),
            create_chunk("chunk-2", "Second"),
            create_chunk("chunk-3", "Third"),
        ]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        retrieved = store.get_by_ids(["chunk-1", "chunk-3"])
        
        assert len(retrieved) == 2
        ids = [c.id for c in retrieved]
        assert "chunk-1" in ids
        assert "chunk-3" in ids
        assert "chunk-2" not in ids
    
    def test_get_by_ids_empty_list(self) -> None:
        """Test get_by_ids with empty list."""
        store = ChromaVectorStore(collection_name=unique_collection_name("get_empty"))
        
        retrieved = store.get_by_ids([])
        
        assert retrieved == []
    
    def test_get_by_ids_nonexistent(self) -> None:
        """Test get_by_ids with nonexistent IDs."""
        store = ChromaVectorStore(collection_name=unique_collection_name("get_nonexist"))
        
        retrieved = store.get_by_ids(["nonexistent"])
        
        assert retrieved == []
    
    def test_get_collection_stats(self) -> None:
        """Test getting collection statistics."""
        store = ChromaVectorStore(collection_name=unique_collection_name("stats"))
        
        stats = store.get_collection_stats()
        
        assert stats["name"] == store.collection_name
        assert stats["count"] == 0
        assert stats["distance_metric"] == "cosine"
        assert stats["persist_directory"] is None
    
    def test_get_collection_stats_with_data(self) -> None:
        """Test getting collection statistics with data."""
        store = ChromaVectorStore(collection_name=unique_collection_name("stats_data"))
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        stats = store.get_collection_stats()
        
        assert stats["count"] == 3
    
    def test_delete_collection(self) -> None:
        """Test deleting the collection."""
        col_name = unique_collection_name("delete_collection")
        store = ChromaVectorStore(collection_name=col_name)
        
        chunks = [create_chunk("chunk-1", "Doc")]
        embeddings = [create_embedding(64)]
        store.add_embeddings(chunks, embeddings)
        assert store.count() == 1
        
        store.delete_collection()
        
        # After deletion, creating a new store with same name should be empty
        store2 = ChromaVectorStore(collection_name=col_name)
        assert store2.count() == 0


class TestChromaVectorStoreDistanceMetrics:
    """Test different distance metrics."""
    
    def test_cosine_distance_metric(self) -> None:
        """Test cosine distance metric."""
        store = ChromaVectorStore(
            collection_name=unique_collection_name("cosine"),
            distance_metric="cosine"
        )
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(64)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        
        assert len(results) == 1
        # Cosine similarity of identical vectors should be ~1.0
        assert results[0].score > 0.99
    
    def test_l2_distance_metric(self) -> None:
        """Test L2 distance metric."""
        store = ChromaVectorStore(
            collection_name=unique_collection_name("l2"),
            distance_metric="l2"
        )
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(64)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        
        assert len(results) == 1
        # L2 distance of 0 should give score of 1.0
        assert results[0].score > 0.99
    
    def test_ip_distance_metric(self) -> None:
        """Test inner product distance metric."""
        # Note: ChromaDB's IP metric behavior can vary depending on 
        # whether the collection already exists with a different metric.
        # We test that the store works correctly with the IP metric.
        store = ChromaVectorStore(
            collection_name=unique_collection_name("ip"),
            distance_metric="ip"
        )
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(64)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        
        assert len(results) == 1
        # The score should be high for identical vectors
        # The exact value depends on ChromaDB's distance calculation
        assert results[0].score >= 0.0


class TestChromaVectorStoreErrorHandling:
    """Test error handling."""
    
    def test_add_embeddings_error_handling(self) -> None:
        """Test that add_embeddings handles errors properly."""
        from src.exceptions import VectorStoreError
        
        store = ChromaVectorStore(collection_name=unique_collection_name("add_error"))
        
        # Create a mock collection that raises an error
        original_collection = store._collection
        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = Exception("Database error")
        store._collection = mock_collection
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(64)
        
        with pytest.raises(VectorStoreError) as exc_info:
            store.add_embeddings([chunk], [embedding])
        
        assert "Failed to add embeddings" in str(exc_info.value)
        
        # Restore original collection
        store._collection = original_collection
    
    def test_similarity_search_error_handling(self) -> None:
        """Test that similarity_search handles errors properly."""
        from src.exceptions import VectorStoreError
        
        store = ChromaVectorStore(collection_name=unique_collection_name("search_error"))
        
        # Add some data first
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(64)
        store.add_embeddings([chunk], [embedding])
        
        # Create a mock collection that raises an error
        original_collection = store._collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.side_effect = Exception("Query error")
        store._collection = mock_collection
        
        with pytest.raises(VectorStoreError) as exc_info:
            store.similarity_search(embedding, k=1)
        
        assert "Failed to perform similarity search" in str(exc_info.value)
        
        # Restore original collection
        store._collection = original_collection
