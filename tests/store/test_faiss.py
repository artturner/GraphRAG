"""Tests for the FAISS vector store implementation.

This module tests the FAISSVectorStore class to ensure it correctly
implements the BaseVectorStore interface and handles all operations properly.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.store.faiss_store import FAISSVectorStore
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


class TestFAISSVectorStoreInit:
    """Test FAISSVectorStore initialization."""
    
    def test_init_with_dimension(self) -> None:
        """Test initialization with just dimension."""
        store = FAISSVectorStore(dimension=128)
        assert store._dimension == 128
        assert store._persist_dir is None
        assert store.count() == 0
    
    def test_init_with_persist_dir(self) -> None:
        """Test initialization with persist directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            assert store._persist_dir == Path(tmpdir)
            assert store.count() == 0
    
    def test_init_normalize_default_true(self) -> None:
        """Test that normalize defaults to True."""
        store = FAISSVectorStore(dimension=32)
        assert store._normalize is True
    
    def test_init_normalize_false(self) -> None:
        """Test initialization with normalize=False."""
        store = FAISSVectorStore(dimension=32, normalize=False)
        assert store._normalize is False


class TestFAISSVectorStoreAddEmbeddings:
    """Test adding embeddings to the store."""
    
    def test_add_single_embedding(self) -> None:
        """Test adding a single chunk with embedding."""
        store = FAISSVectorStore(dimension=64)
        chunk = create_chunk("chunk-1", "Hello world")
        embedding = create_embedding(64, seed=1)
        
        count = store.add_embeddings([chunk], [embedding])
        
        assert count == 1
        assert store.count() == 1
    
    def test_add_multiple_embeddings(self) -> None:
        """Test adding multiple chunks with embeddings."""
        store = FAISSVectorStore(dimension=64)
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
        store = FAISSVectorStore(dimension=64)
        
        count = store.add_embeddings([], [])
        
        assert count == 0
        assert store.count() == 0
    
    def test_add_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched chunk/embedding lengths raises ValueError."""
        store = FAISSVectorStore(dimension=64)
        chunks = [create_chunk("chunk-1", "Test")]
        embeddings = [create_embedding(64), create_embedding(64)]
        
        with pytest.raises(ValueError) as exc_info:
            store.add_embeddings(chunks, embeddings)
        
        assert "must match" in str(exc_info.value)
    
    def test_add_wrong_dimension_raises_error(self) -> None:
        """Test that wrong embedding dimension raises VectorStoreError."""
        from src.exceptions import VectorStoreError
        store = FAISSVectorStore(dimension=64)
        chunk = create_chunk("chunk-1", "Test")
        wrong_embedding = create_embedding(32)  # Wrong dimension
        
        with pytest.raises(VectorStoreError) as exc_info:
            store.add_embeddings([chunk], [wrong_embedding])
        
        assert "dimension mismatch" in str(exc_info.value)
    
    def test_add_incremental(self) -> None:
        """Test adding embeddings incrementally."""
        store = FAISSVectorStore(dimension=64)
        
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
        store = FAISSVectorStore(dimension=64)
        
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


class TestFAISSVectorStoreSimilaritySearch:
    """Test similarity search functionality."""
    
    def test_search_returns_correct_results(self) -> None:
        """Test that search returns correct results."""
        store = FAISSVectorStore(dimension=64)
        
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
        store = FAISSVectorStore(dimension=64)
        
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
        store = FAISSVectorStore(dimension=64)
        
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
        store = FAISSVectorStore(dimension=64)
        
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
    
    def test_search_wrong_dimension_raises_error(self) -> None:
        """Test that search with wrong dimension raises error."""
        from src.exceptions import VectorStoreError
        store = FAISSVectorStore(dimension=64)
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(64, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        wrong_query = create_embedding(32)  # Wrong dimension
        
        with pytest.raises(VectorStoreError) as exc_info:
            store.similarity_search(wrong_query, k=1)
        
        assert "dimension mismatch" in str(exc_info.value)


class TestFAISSVectorStoreEmptyBehavior:
    """Test behavior with empty store."""
    
    def test_search_empty_store_returns_empty_list(self) -> None:
        """Test that searching an empty store returns empty list."""
        store = FAISSVectorStore(dimension=64)
        query = create_embedding(64)
        
        results = store.similarity_search(query, k=5)
        
        assert results == []
    
    def test_count_empty_store(self) -> None:
        """Test count on empty store."""
        store = FAISSVectorStore(dimension=64)
        assert store.count() == 0
    
    def test_delete_empty_store(self) -> None:
        """Test delete on empty store."""
        store = FAISSVectorStore(dimension=64)
        deleted = store.delete(["nonexistent"])
        assert deleted == 0
    
    def test_clear_empty_store(self) -> None:
        """Test clear on empty store."""
        store = FAISSVectorStore(dimension=64)
        store.clear()  # Should not raise
        assert store.count() == 0


class TestFAISSVectorStoreDelete:
    """Test delete functionality."""
    
    def test_delete_single_chunk(self) -> None:
        """Test deleting a single chunk."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        assert store.count() == 3
        
        deleted = store.delete(["chunk-1"])
        
        assert deleted == 1
        assert store.count() == 2
    
    def test_delete_multiple_chunks(self) -> None:
        """Test deleting multiple chunks."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(5)]
        embeddings = [create_embedding(64, seed=i) for i in range(5)]
        store.add_embeddings(chunks, embeddings)
        assert store.count() == 5
        
        deleted = store.delete(["chunk-1", "chunk-3", "chunk-4"])
        
        assert deleted == 3
        assert store.count() == 2
    
    def test_delete_nonexistent_chunk(self) -> None:
        """Test deleting a chunk that doesn't exist."""
        store = FAISSVectorStore(dimension=64)
        
        chunk = create_chunk("chunk-1", "Doc")
        embedding = create_embedding(64, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        deleted = store.delete(["nonexistent"])
        
        assert deleted == 0
        assert store.count() == 1
    
    def test_delete_partial_mixed(self) -> None:
        """Test deleting mix of existing and non-existing chunks."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        deleted = store.delete(["chunk-1", "nonexistent", "chunk-2"])
        
        assert deleted == 2
        assert store.count() == 1
    
    def test_delete_empty_list(self) -> None:
        """Test deleting with empty list."""
        store = FAISSVectorStore(dimension=64)
        
        chunk = create_chunk("chunk-1", "Doc")
        embedding = create_embedding(64, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        deleted = store.delete([])
        
        assert deleted == 0
        assert store.count() == 1
    
    def test_delete_and_search(self) -> None:
        """Test that deleted chunks don't appear in search."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = create_orthogonal_embeddings(64, 3)
        store.add_embeddings(chunks, embeddings)
        
        # Delete chunk-1
        store.delete(["chunk-1"])
        
        # Search should not return chunk-1
        results = store.similarity_search(embeddings[1], k=3)
        chunk_ids = [r.chunk_id for r in results]
        
        assert "chunk-1" not in chunk_ids
        assert len(results) == 2


class TestFAISSVectorStoreCount:
    """Test count functionality."""
    
    def test_count_after_add(self) -> None:
        """Test count after adding embeddings."""
        store = FAISSVectorStore(dimension=64)
        
        assert store.count() == 0
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        assert store.count() == 3
    
    def test_count_after_delete(self) -> None:
        """Test count after deleting."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        store.delete(["chunk-1"])
        assert store.count() == 2
        
        store.delete(["chunk-0", "chunk-2"])
        assert store.count() == 0
    
    def test_count_after_clear(self) -> None:
        """Test count after clearing."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        store.clear()
        assert store.count() == 0


class TestFAISSVectorStoreClear:
    """Test clear functionality."""
    
    def test_clear_removes_all_chunks(self) -> None:
        """Test that clear removes all chunks."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(5)]
        embeddings = [create_embedding(64, seed=i) for i in range(5)]
        store.add_embeddings(chunks, embeddings)
        
        store.clear()
        
        assert store.count() == 0
    
    def test_clear_allows_adding_again(self) -> None:
        """Test that store can be used after clearing."""
        store = FAISSVectorStore(dimension=64)
        
        # Add and clear
        chunks1 = [create_chunk("chunk-1", "First")]
        embeddings1 = [create_embedding(64, seed=1)]
        store.add_embeddings(chunks1, embeddings1)
        store.clear()
        
        # Add again
        chunks2 = [create_chunk("chunk-2", "Second")]
        embeddings2 = [create_embedding(64, seed=2)]
        store.add_embeddings(chunks2, embeddings2)
        
        assert store.count() == 1
        results = store.similarity_search(embeddings2[0], k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-2"


class TestFAISSVectorStorePersistence:
    """Test persistence functionality."""
    
    def test_save_without_persist_dir_raises_error(self) -> None:
        """Test that save without persist_dir raises error."""
        from src.exceptions import VectorStoreError
        store = FAISSVectorStore(dimension=64)
        
        with pytest.raises(VectorStoreError) as exc_info:
            store.save()
        
        assert "persist_dir" in str(exc_info.value)
    
    def test_save_and_load(self) -> None:
        """Test saving and loading the index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate store
            store1 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
            embeddings = create_orthogonal_embeddings(64, 3)
            store1.add_embeddings(chunks, embeddings)
            
            # Save
            store1.save()
            
            # Verify files exist
            assert (Path(tmpdir) / "index.faiss").exists()
            assert (Path(tmpdir) / "metadata.json").exists()
            assert (Path(tmpdir) / "embeddings.npz").exists()
            
            # Load into new store
            store2 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            
            assert store2.count() == 3
            
            # Verify search works
            results = store2.similarity_search(embeddings[0], k=1)
            assert results[0].chunk_id == "chunk-0"
    
    def test_load_nonexistent_directory(self) -> None:
        """Test loading from nonexistent directory starts empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "nonexistent"
            
            store = FAISSVectorStore(dimension=64, persist_dir=persist_path)
            
            assert store.count() == 0
    
    def test_persistence_preserves_metadata(self) -> None:
        """Test that metadata is preserved after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store with metadata
            store1 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            chunk = Chunk(
                id="chunk-1",
                document_id="doc-1",
                content="Test content",
                start_idx=0,
                end_idx=12,
                metadata={"author": "test", "page": 1},
            )
            embedding = create_embedding(64, seed=1)
            store1.add_embeddings([chunk], [embedding])
            store1.save()
            
            # Load and verify
            store2 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            results = store2.similarity_search(embedding, k=1)
            
            assert results[0].chunk.metadata["author"] == "test"
            assert results[0].chunk.metadata["page"] == 1
    
    def test_persistence_after_delete(self) -> None:
        """Test persistence after deletions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            store1 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(5)]
            embeddings = [create_embedding(64, seed=i) for i in range(5)]
            store1.add_embeddings(chunks, embeddings)
            store1.save()
            
            # Delete some and save again
            store1.delete(["chunk-1", "chunk-3"])
            store1.save()
            
            # Load and verify
            store2 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            assert store2.count() == 3
            
            # Verify deleted chunks are not present
            results = store2.similarity_search(embeddings[0], k=5)
            chunk_ids = {r.chunk_id for r in results}
            assert "chunk-1" not in chunk_ids
            assert "chunk-3" not in chunk_ids
    
    def test_incremental_add_and_save(self) -> None:
        """Test incremental additions with save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First batch
            store1 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            chunks1 = [create_chunk("chunk-1", "First")]
            embeddings1 = [create_embedding(64, seed=1)]
            store1.add_embeddings(chunks1, embeddings1)
            store1.save()
            
            # Second batch (load, add, save)
            store2 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            assert store2.count() == 1
            chunks2 = [create_chunk("chunk-2", "Second")]
            embeddings2 = [create_embedding(64, seed=2)]
            store2.add_embeddings(chunks2, embeddings2)
            store2.save()
            
            # Verify both chunks
            store3 = FAISSVectorStore(dimension=64, persist_dir=tmpdir)
            assert store3.count() == 2


class TestFAISSVectorStoreNormalization:
    """Test vector normalization behavior."""
    
    def test_normalized_vectors_cosine_similarity(self) -> None:
        """Test that normalized vectors produce cosine-like similarity."""
        store = FAISSVectorStore(dimension=64, normalize=True)
        
        # Create two similar vectors
        base = np.random.randn(64).astype(np.float32)
        base = base / np.linalg.norm(base)
        
        # Slightly modified version
        modified = base + np.random.randn(64).astype(np.float32) * 0.1
        modified = modified / np.linalg.norm(modified)
        
        chunks = [
            create_chunk("chunk-1", "First"),
            create_chunk("chunk-2", "Second"),
        ]
        embeddings = [base.tolist(), modified.tolist()]
        store.add_embeddings(chunks, embeddings)
        
        # Search with base should find both, with chunk-1 first
        results = store.similarity_search(base.tolist(), k=2)
        
        assert results[0].chunk_id == "chunk-1"
        assert results[0].score > results[1].score
    
    def test_unnormalized_vectors(self) -> None:
        """Test store with normalization disabled."""
        store = FAISSVectorStore(dimension=64, normalize=False)
        
        # Create normalized vectors manually
        base = np.random.randn(64).astype(np.float32)
        base = base / np.linalg.norm(base)
        
        chunk = create_chunk("chunk-1", "Test")
        store.add_embeddings([chunk], [base.tolist()])
        
        # Search should still work
        results = store.similarity_search(base.tolist(), k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"


class TestFAISSVectorStoreEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_large_k_value(self) -> None:
        """Test search with k larger than store size."""
        store = FAISSVectorStore(dimension=64)
        
        chunks = [create_chunk(f"chunk-{i}", f"Doc {i}") for i in range(3)]
        embeddings = [create_embedding(64, seed=i) for i in range(3)]
        store.add_embeddings(chunks, embeddings)
        
        results = store.similarity_search(embeddings[0], k=100)
        
        assert len(results) == 3
    
    def test_very_small_dimension(self) -> None:
        """Test with very small dimension."""
        store = FAISSVectorStore(dimension=2)
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = [0.707, 0.707]  # Normalized 2D vector
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        assert len(results) == 1
    
    def test_large_dimension(self) -> None:
        """Test with large dimension."""
        dimension = 1536  # Common embedding size
        store = FAISSVectorStore(dimension=dimension)
        
        chunk = create_chunk("chunk-1", "Test")
        embedding = create_embedding(dimension, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        assert len(results) == 1
    
    def test_score_clamping_negative(self) -> None:
        """Test that negative scores are clamped to 0."""
        store = FAISSVectorStore(dimension=64, normalize=True)
        
        # Create orthogonal vectors (will have negative inner product)
        embeddings = create_orthogonal_embeddings(64, 2)
        
        chunks = [
            create_chunk("chunk-1", "First"),
            create_chunk("chunk-2", "Second"),
        ]
        store.add_embeddings(chunks, embeddings)
        
        # Search with opposite vector
        opposite = [-x for x in embeddings[0]]
        results = store.similarity_search(opposite, k=2)
        
        # All scores should be >= 0
        for result in results:
            assert result.score >= 0.0
    
    def test_chunk_with_empty_metadata(self) -> None:
        """Test chunk with empty metadata."""
        store = FAISSVectorStore(dimension=64)
        
        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="Test",
            start_idx=0,
            end_idx=4,
            metadata={},  # Empty metadata
        )
        embedding = create_embedding(64, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        assert results[0].metadata == {}
    
    def test_chunk_with_complex_metadata(self) -> None:
        """Test chunk with complex metadata."""
        store = FAISSVectorStore(dimension=64)
        
        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="Test",
            start_idx=0,
            end_idx=4,
            metadata={
                "author": "John Doe",
                "tags": ["tag1", "tag2"],
                "nested": {"key": "value"},
                "number": 42,
            },
        )
        embedding = create_embedding(64, seed=1)
        store.add_embeddings([chunk], [embedding])
        
        results = store.similarity_search(embedding, k=1)
        assert results[0].metadata["author"] == "John Doe"
        assert results[0].metadata["tags"] == ["tag1", "tag2"]
        assert results[0].metadata["nested"]["key"] == "value"
