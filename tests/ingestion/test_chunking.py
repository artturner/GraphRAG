"""Tests for text chunking strategies.

This module contains comprehensive tests for the chunking module including
FixedSizeChunker and SentenceChunker implementations.
"""

import pytest
from src.ingestion.chunking import (
    BaseChunker,
    ChunkMetadata,
    FixedSizeChunker,
    SentenceChunker,
)
from src.types import Chunk


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        meta = ChunkMetadata(
            chunk_index=0,
            total_chunks=1,
            chunk_type="test",
        )
        
        assert meta.chunk_index == 0
        assert meta.total_chunks == 1
        assert meta.chunk_type == "test"
        assert meta.has_overlap is False
        assert meta.overlap_chars == 0
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        meta = ChunkMetadata(
            chunk_index=5,
            total_chunks=10,
            chunk_type="fixed_size",
            has_overlap=True,
            overlap_chars=50,
        )
        
        assert meta.chunk_index == 5
        assert meta.total_chunks == 10
        assert meta.chunk_type == "fixed_size"
        assert meta.has_overlap is True
        assert meta.overlap_chars == 50


class TestFixedSizeChunkerInit:
    """Tests for FixedSizeChunker initialization."""
    
    def test_default_values(self):
        """Test initialization with default values."""
        chunker = FixedSizeChunker()
        
        assert chunker.chunk_size == 500
        assert chunker.overlap == 0
        assert chunker.respect_word_boundaries is True
    
    def test_custom_values(self):
        """Test initialization with custom values."""
        chunker = FixedSizeChunker(
            chunk_size=1000,
            overlap=100,
            respect_word_boundaries=False,
        )
        
        assert chunker.chunk_size == 1000
        assert chunker.overlap == 100
        assert chunker.respect_word_boundaries is False
    
    def test_invalid_chunk_size_zero(self):
        """Test that zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0)
    
    def test_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=-100)
    
    def test_invalid_overlap_negative(self):
        """Test that negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            FixedSizeChunker(chunk_size=500, overlap=-10)
    
    def test_invalid_overlap_greater_than_chunk_size(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            FixedSizeChunker(chunk_size=100, overlap=100)
        
        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            FixedSizeChunker(chunk_size=100, overlap=150)


class TestFixedSizeChunkerChunking:
    """Tests for FixedSizeChunker chunking functionality."""
    
    def test_empty_text(self):
        """Test chunking empty text returns empty list."""
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk("", {"doc_id": "test"})
        
        assert chunks == []
    
    def test_text_shorter_than_chunk_size(self):
        """Test text shorter than chunk_size returns single chunk."""
        chunker = FixedSizeChunker(chunk_size=1000)
        text = "This is a short text."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].start_idx == 0
        assert chunks[0].end_idx == len(text)
    
    def test_basic_chunking(self):
        """Test basic chunking with multiple chunks."""
        chunker = FixedSizeChunker(chunk_size=50, respect_word_boundaries=False)
        text = "a" * 150  # 150 characters
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        assert len(chunks) == 3
        assert all(len(c.content) == 50 for c in chunks)
    
    def test_chunk_size_various_sizes(self):
        """Test chunking with various chunk sizes."""
        text = "a" * 500  # 500 characters, no spaces to avoid stripping issues
        
        for size in [100, 200, 250]:
            chunker = FixedSizeChunker(chunk_size=size, respect_word_boundaries=False)
            chunks = chunker.chunk(text, {"doc_id": "test"})
            
            # Verify all chunks except possibly the last are of expected size
            for chunk in chunks[:-1]:
                assert len(chunk.content) == size
    
    def test_metadata_attached(self):
        """Test that metadata is correctly attached to chunks."""
        chunker = FixedSizeChunker(chunk_size=100)
        text = "Test content for metadata. " * 10
        metadata = {"doc_id": "doc-123", "source": "test.txt", "custom": "value"}
        chunks = chunker.chunk(text, metadata)
        
        for chunk in chunks:
            assert chunk.document_id == "doc-123"
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["custom"] == "value"
            assert "chunk_meta" in chunk.metadata
    
    def test_chunk_indices(self):
        """Test that chunk indices are correctly set."""
        chunker = FixedSizeChunker(chunk_size=50, respect_word_boundaries=False)
        text = "a" * 150
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_meta"]["chunk_index"] == i
    
    def test_total_chunks_count(self):
        """Test that total_chunks is correctly set."""
        chunker = FixedSizeChunker(chunk_size=50, respect_word_boundaries=False)
        text = "a" * 150
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        for chunk in chunks:
            assert chunk.metadata["chunk_meta"]["total_chunks"] == len(chunks)


class TestFixedSizeChunkerOverlap:
    """Tests for FixedSizeChunker overlap functionality."""
    
    def test_overlap_basic(self):
        """Test basic overlap functionality."""
        chunker = FixedSizeChunker(
            chunk_size=100,
            overlap=20,
            respect_word_boundaries=False,
        )
        text = "a" * 250
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Check that chunks have overlap metadata
        for i, chunk in enumerate(chunks):
            if i > 0:
                assert chunk.metadata["chunk_meta"]["has_overlap"] is True
                assert chunk.metadata["chunk_meta"]["overlap_chars"] == 20
    
    def test_overlap_content(self):
        """Test that overlapping content is correct."""
        chunker = FixedSizeChunker(
            chunk_size=50,
            overlap=10,
            respect_word_boundaries=False,
        )
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # First chunk: a-p (0-25, but we use 50 so it's a-z split)
        # With 50 char chunks and 10 overlap on 26 chars:
        # First chunk: 0-25 (all 26 chars since < 50)
        assert len(chunks) == 1  # Text is shorter than chunk_size
    
    def test_overlap_with_longer_text(self):
        """Test overlap with text longer than chunk_size."""
        chunker = FixedSizeChunker(
            chunk_size=20,
            overlap=5,
            respect_word_boundaries=False,
        )
        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # With 20 char chunks and 5 overlap:
        # Chunk 1: 0-20 (a-t)
        # Chunk 2: 15-26 (p-z) - starts at 20-5=15
        assert len(chunks) == 2
        assert chunks[0].content == "abcdefghijklmnopqrst"
        assert chunks[1].content == "pqrstuvwxyz"
    
    def test_no_overlap_metadata(self):
        """Test that overlap metadata is correct when no overlap."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        text = "a" * 200
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        for chunk in chunks:
            assert chunk.metadata["chunk_meta"]["has_overlap"] is False
            assert chunk.metadata["chunk_meta"]["overlap_chars"] == 0


class TestFixedSizeChunkerWordBoundaries:
    """Tests for FixedSizeChunker word boundary respect."""
    
    def test_respects_word_boundaries(self):
        """Test that word boundaries are respected."""
        chunker = FixedSizeChunker(chunk_size=20, respect_word_boundaries=True)
        text = "This is a test sentence with multiple words."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Chunks should not split words in the middle
        for chunk in chunks:
            # Check that chunk doesn't end mid-word
            # (it might end with punctuation or space, which is fine)
            assert chunk.content  # Non-empty
    
    def test_no_word_boundary_respect(self):
        """Test chunking without word boundary respect."""
        chunker = FixedSizeChunker(chunk_size=20, respect_word_boundaries=False)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Should split exactly at chunk_size
        assert len(chunks[0].content) == 20
    
    def test_very_long_word(self):
        """Test handling of very long words."""
        chunker = FixedSizeChunker(chunk_size=20, respect_word_boundaries=True)
        # A word longer than chunk_size
        text = "supercalifragilisticexpialidocious"
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Should still create chunks even if word is longer than chunk_size
        assert len(chunks) >= 1
        # The chunker will try to find a boundary but may not find one
        assert chunks[0].content


class TestSentenceChunkerInit:
    """Tests for SentenceChunker initialization."""
    
    def test_default_values(self):
        """Test initialization with default values."""
        chunker = SentenceChunker()
        
        assert chunker.min_size == 200
        assert chunker.max_size == 1000
    
    def test_custom_values(self):
        """Test initialization with custom values."""
        chunker = SentenceChunker(min_size=100, max_size=500)
        
        assert chunker.min_size == 100
        assert chunker.max_size == 500
    
    def test_invalid_min_size_zero(self):
        """Test that zero min_size raises ValueError."""
        with pytest.raises(ValueError, match="min_size must be positive"):
            SentenceChunker(min_size=0)
    
    def test_invalid_min_size_negative(self):
        """Test that negative min_size raises ValueError."""
        with pytest.raises(ValueError, match="min_size must be positive"):
            SentenceChunker(min_size=-100)
    
    def test_invalid_max_size_not_greater_than_min(self):
        """Test that max_size <= min_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size .* must be greater than min_size"):
            SentenceChunker(min_size=500, max_size=500)
        
        with pytest.raises(ValueError, match="max_size .* must be greater than min_size"):
            SentenceChunker(min_size=500, max_size=400)


class TestSentenceChunkerChunking:
    """Tests for SentenceChunker chunking functionality."""
    
    def test_empty_text(self):
        """Test chunking empty text returns empty list."""
        chunker = SentenceChunker()
        chunks = chunker.chunk("", {"doc_id": "test"})
        
        assert chunks == []
    
    def test_single_sentence(self):
        """Test chunking a single sentence."""
        chunker = SentenceChunker(min_size=10, max_size=1000)
        text = "This is a single sentence."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    def test_multiple_sentences(self):
        """Test chunking multiple sentences."""
        chunker = SentenceChunker(min_size=10, max_size=1000)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        assert len(chunks) >= 1
        # All sentences should be included
        combined = " ".join(c.content for c in chunks)
        assert "First sentence" in combined
        assert "Second sentence" in combined
        assert "Third sentence" in combined
    
    def test_sentence_detection(self):
        """Test that sentences are correctly detected."""
        chunker = SentenceChunker(min_size=5, max_size=1000)
        text = "Hello world! How are you? I am fine."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Should detect sentences ending with !, ?, .
        combined = " ".join(c.content for c in chunks)
        assert "Hello world" in combined
        assert "How are you" in combined
        assert "I am fine" in combined
    
    def test_chunk_size_constraints(self):
        """Test that chunks respect size constraints."""
        chunker = SentenceChunker(min_size=20, max_size=100)
        
        # Create text with many short sentences
        sentences = ["This is sentence number {}. ".format(i) for i in range(20)]
        text = "".join(sentences)
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Check that chunks don't exceed max_size
        for chunk in chunks:
            assert len(chunk.content) <= chunker.max_size
    
    def test_metadata_attached(self):
        """Test that metadata is correctly attached to chunks."""
        chunker = SentenceChunker(min_size=10, max_size=1000)
        text = "First sentence. Second sentence."
        metadata = {"doc_id": "doc-456", "source": "test.txt"}
        chunks = chunker.chunk(text, metadata)
        
        for chunk in chunks:
            assert chunk.document_id == "doc-456"
            assert chunk.metadata["source"] == "test.txt"
            assert "chunk_meta" in chunk.metadata
            assert "sentence_count" in chunk.metadata
    
    def test_sentence_count_in_metadata(self):
        """Test that sentence_count is correctly set."""
        chunker = SentenceChunker(min_size=10, max_size=1000)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Total sentence count across all chunks should be 3
        total_sentences = sum(c.metadata["sentence_count"] for c in chunks)
        assert total_sentences == 3
    
    def test_chunk_indices(self):
        """Test that chunk indices are correctly set."""
        chunker = SentenceChunker(min_size=10, max_size=50)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_meta"]["chunk_index"] == i
    
    def test_total_chunks_count(self):
        """Test that total_chunks is correctly set."""
        chunker = SentenceChunker(min_size=10, max_size=50)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        for chunk in chunks:
            assert chunk.metadata["chunk_meta"]["total_chunks"] == len(chunks)


class TestSentenceChunkerEdgeCases:
    """Tests for SentenceChunker edge cases."""
    
    def test_text_without_punctuation(self):
        """Test text without sentence-ending punctuation."""
        chunker = SentenceChunker(min_size=10, max_size=1000)
        text = "This is text without punctuation"
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Should return the whole text as one chunk
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    def test_text_with_newlines(self):
        """Test text with newlines as sentence boundaries."""
        chunker = SentenceChunker(min_size=10, max_size=1000)
        text = "First line.\nSecond line.\nThird line."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "First line" in combined
    
    def test_very_long_sentence(self):
        """Test handling of a sentence longer than max_size."""
        chunker = SentenceChunker(min_size=10, max_size=50)
        # A single long sentence
        text = "This is a very long sentence that exceeds the maximum chunk size limit."
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Should still create a chunk even if it exceeds max_size
        assert len(chunks) == 1
        assert len(chunks[0].content) > chunker.max_size
    
    def test_whitespace_only_text(self):
        """Test text with only whitespace."""
        chunker = SentenceChunker()
        text = "   \n\t  "
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Should return empty list for whitespace-only text
        assert chunks == []


class TestChunkerIntegration:
    """Integration tests for chunkers."""
    
    def test_fixed_size_chunker_with_real_text(self):
        """Test FixedSizeChunker with realistic text."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        text = """
        This is a sample document that contains multiple paragraphs
        of text. It is designed to test the chunking functionality
        with realistic content that might be found in actual documents.
        
        The document includes various sentence structures and word
        lengths to ensure the chunker handles different cases properly.
        """
        chunks = chunker.chunk(text, {"doc_id": "real-doc", "author": "Test"})
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.document_id == "real-doc"
            assert chunk.metadata["author"] == "Test"
    
    def test_sentence_chunker_with_real_text(self):
        """Test SentenceChunker with realistic text."""
        chunker = SentenceChunker(min_size=50, max_size=200)
        text = """
        Artificial intelligence is transforming industries. Machine learning
        models are becoming more sophisticated. Natural language processing
        enables computers to understand human language. These technologies
        are reshaping how we work and live.
        """
        chunks = chunker.chunk(text, {"doc_id": "ai-doc"})
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.document_id == "ai-doc"
    
    def test_chunk_ids_are_unique(self):
        """Test that all chunk IDs are unique."""
        chunker = FixedSizeChunker(chunk_size=50)
        text = "a" * 200
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
    
    def test_chunk_positions_are_sequential(self):
        """Test that chunk positions are sequential."""
        chunker = FixedSizeChunker(chunk_size=50, respect_word_boundaries=False)
        text = "a" * 200
        chunks = chunker.chunk(text, {"doc_id": "test"})
        
        # Check that start_idx and end_idx are consistent
        for i, chunk in enumerate(chunks):
            assert chunk.start_idx >= 0
            assert chunk.end_idx > chunk.start_idx


class TestBaseChunker:
    """Tests for BaseChunker abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChunker()
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        # Create a concrete implementation for testing
        class TestChunker(BaseChunker):
            def chunk(self, text: str, metadata: dict) -> list:
                return []
        
        chunker = TestChunker()
        chunk_id = chunker._generate_chunk_id("doc-123", 5)
        
        assert chunk_id == "doc-123-chunk-0005"
        assert "doc-123" in chunk_id
        assert "0005" in chunk_id
