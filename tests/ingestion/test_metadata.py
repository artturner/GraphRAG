"""Tests for metadata extraction module.

This module tests the ChunkMetadataExtractor class and related functions
for extracting metadata from document chunks.
"""

import json
from datetime import datetime

import pytest

from src.ingestion.metadata import (
    ChunkMetadataExtractor,
    extract_context,
    extract_position_info,
    generate_chunk_id,
)
from src.types import Chunk, Document, DocumentMetadata, DocumentType


# Fixtures
@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    content = (
        "This is the beginning of the document. "
        "It contains multiple sentences. "
        "This is the middle section of the document. "
        "We are adding more content here. "
        "This is the end of the document."
    )
    return Document(
        id="doc_123",
        content=content,
        source="/path/to/document.txt",
        document_type=DocumentType.TXT,
        metadata={"category": "test", "version": "1.0"},
        full_metadata=DocumentMetadata(
            author="Test Author",
            title="Test Document",
            tags=["test", "sample"],
        ),
    )


@pytest.fixture
def sample_chunk(sample_document: Document) -> Chunk:
    """Create a sample chunk for testing."""
    return Chunk(
        id="doc_123_chunk_0",
        document_id="doc_123",
        content="This is the middle section of the document.",
        start_idx=62,
        end_idx=102,
        metadata={
            "page": 1,
            "paragraph": 2,
            "chunk_meta": {
                "chunk_index": 0,
                "total_chunks": 3,
                "chunk_type": "fixed_size",
                "has_overlap": False,
                "overlap_chars": 0,
            },
        },
    )


class TestGenerateChunkId:
    """Tests for generate_chunk_id function."""

    def test_generates_deterministic_id(self) -> None:
        """Test that chunk ID generation is deterministic."""
        id1 = generate_chunk_id("doc_123", 0)
        id2 = generate_chunk_id("doc_123", 0)
        assert id1 == id2
        assert id1 == "doc_123_chunk_0"

    def test_different_document_ids(self) -> None:
        """Test that different document IDs produce different chunk IDs."""
        id1 = generate_chunk_id("doc_123", 0)
        id2 = generate_chunk_id("doc_456", 0)
        assert id1 != id2

    def test_different_chunk_indices(self) -> None:
        """Test that different chunk indices produce different chunk IDs."""
        id1 = generate_chunk_id("doc_123", 0)
        id2 = generate_chunk_id("doc_123", 1)
        id3 = generate_chunk_id("doc_123", 5)
        assert id1 != id2
        assert id2 != id3
        assert id1 == "doc_123_chunk_0"
        assert id2 == "doc_123_chunk_1"
        assert id3 == "doc_123_chunk_5"

    def test_various_document_id_formats(self) -> None:
        """Test chunk ID generation with various document ID formats."""
        # UUID format
        uuid_id = generate_chunk_id("550e8400-e29b-41d4-a716-446655440000", 0)
        assert uuid_id == "550e8400-e29b-41d4-a716-446655440000_chunk_0"

        # Simple format
        simple_id = generate_chunk_id("doc1", 10)
        assert simple_id == "doc1_chunk_10"

        # With underscores
        underscore_id = generate_chunk_id("my_document_v2", 3)
        assert underscore_id == "my_document_v2_chunk_3"


class TestExtractPositionInfo:
    """Tests for extract_position_info function."""

    def test_extracts_basic_position_info(self, sample_chunk: Chunk) -> None:
        """Test extraction of basic position information."""
        position_info = extract_position_info(sample_chunk)

        assert position_info["start_char"] == 62
        assert position_info["end_char"] == 102
        assert position_info["char_count"] == len(sample_chunk.content)

    def test_extracts_page_and_paragraph(self, sample_chunk: Chunk) -> None:
        """Test extraction of page and paragraph information."""
        position_info = extract_position_info(sample_chunk)

        assert position_info["page"] == 1
        assert position_info["paragraph"] == 2

    def test_extracts_chunk_metadata(self, sample_chunk: Chunk) -> None:
        """Test extraction of chunk metadata."""
        position_info = extract_position_info(sample_chunk)

        assert position_info["chunk_index"] == 0
        assert position_info["total_chunks"] == 3
        assert position_info["chunk_type"] == "fixed_size"
        assert position_info["has_overlap"] is False
        assert position_info["overlap_chars"] == 0

    def test_handles_missing_metadata(self) -> None:
        """Test handling of chunk with missing metadata."""
        chunk = Chunk(
            id="test_chunk",
            document_id="doc_1",
            content="Test content",
            start_idx=0,
            end_idx=12,
            metadata={},
        )
        position_info = extract_position_info(chunk)

        assert position_info["start_char"] == 0
        assert position_info["end_char"] == 12
        assert position_info["page"] is None
        assert position_info["paragraph"] is None
        assert position_info["chunk_index"] is None
        assert position_info["total_chunks"] is None

    def test_handles_overlap_metadata(self) -> None:
        """Test handling of chunk with overlap information."""
        chunk = Chunk(
            id="test_chunk",
            document_id="doc_1",
            content="Test content",
            start_idx=50,
            end_idx=100,
            metadata={
                "chunk_meta": {
                    "chunk_index": 1,
                    "total_chunks": 5,
                    "chunk_type": "fixed_size",
                    "has_overlap": True,
                    "overlap_chars": 50,
                },
            },
        )
        position_info = extract_position_info(chunk)

        assert position_info["has_overlap"] is True
        assert position_info["overlap_chars"] == 50


class TestExtractContext:
    """Tests for extract_context function."""

    def test_extracts_context_before_and_after(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test extraction of context before and after chunk."""
        context = extract_context(sample_chunk, sample_document)

        assert "before" in context
        assert "after" in context
        assert len(context["before"]) > 0
        assert len(context["after"]) > 0

    def test_includes_document_info(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that document info is included in context."""
        context = extract_context(sample_chunk, sample_document)

        assert context["document_source"] == sample_document.source
        assert context["document_type"] == sample_document.document_type.value
        assert context["document_id"] == sample_document.id

    def test_context_at_document_start(self, sample_document: Document) -> None:
        """Test context extraction for chunk at document start."""
        chunk = Chunk(
            id="doc_123_chunk_0",
            document_id="doc_123",
            content=sample_document.content[:50],
            start_idx=0,
            end_idx=50,
            metadata={},
        )
        context = extract_context(chunk, sample_document)

        # No text before, so before should be empty or just ellipsis
        assert context["before"] == "" or context["before"].startswith("...")
        assert len(context["after"]) > 0

    def test_context_at_document_end(self, sample_document: Document) -> None:
        """Test context extraction for chunk at document end."""
        content_len = len(sample_document.content)
        chunk = Chunk(
            id="doc_123_chunk_last",
            document_id="doc_123",
            content=sample_document.content[-50:],
            start_idx=content_len - 50,
            end_idx=content_len,
            metadata={},
        )
        context = extract_context(chunk, sample_document)

        assert len(context["before"]) > 0
        # After should be empty since we're at the end
        assert context["after"] == "" or not context["after"].endswith("...")

    def test_context_with_none_document_type(self) -> None:
        """Test context extraction when document type is None."""
        doc = Document(
            id="doc_1",
            content="This is a test document with some content.",
            source="/path/to/doc.txt",
            document_type=None,
        )
        chunk = Chunk(
            id="doc_1_chunk_0",
            document_id="doc_1",
            content="test document",
            start_idx=10,
            end_idx=23,
            metadata={},
        )
        context = extract_context(chunk, doc)

        assert context["document_type"] is None


class TestChunkMetadataExtractor:
    """Tests for ChunkMetadataExtractor class."""

    def test_extract_returns_comprehensive_metadata(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that extract returns comprehensive metadata."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        assert "chunk_id" in metadata
        assert "document_id" in metadata
        assert "position" in metadata
        assert "context" in metadata
        assert "document_metadata" in metadata
        assert "content_length" in metadata
        assert "content_hash" in metadata

    def test_extract_includes_correct_ids(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that extract includes correct chunk and document IDs."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        assert metadata["chunk_id"] == sample_chunk.id
        assert metadata["document_id"] == sample_chunk.document_id

    def test_extract_includes_position_info(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that extract includes position information."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        position = metadata["position"]
        assert position["start_char"] == sample_chunk.start_idx
        assert position["end_char"] == sample_chunk.end_idx
        # Relative position should be calculated
        assert 0.0 <= position["position"] <= 1.0

    def test_extract_includes_context(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that extract includes context information."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        context = metadata["context"]
        assert "before" in context
        assert "after" in context

    def test_extract_includes_document_metadata(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that extract includes document-level metadata."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        doc_meta = metadata["document_metadata"]
        assert doc_meta["source"] == sample_document.source
        assert doc_meta["author"] == sample_document.full_metadata.author
        assert doc_meta["title"] == sample_document.full_metadata.title
        assert doc_meta["tags"] == sample_document.full_metadata.tags

    def test_custom_context_window(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that custom context window is respected."""
        extractor = ChunkMetadataExtractor(context_window=50)
        metadata = extractor.extract(sample_chunk, sample_document)

        context = metadata["context"]
        # Context should be limited to ~50 chars (plus ellipsis)
        before_without_ellipsis = context["before"].replace("...", "")
        after_without_ellipsis = context["after"].replace("...", "")
        assert len(before_without_ellipsis) <= 50
        assert len(after_without_ellipsis) <= 50

    def test_invalid_context_window_raises(self) -> None:
        """Test that negative context window raises ValueError."""
        with pytest.raises(ValueError, match="context_window must be non-negative"):
            ChunkMetadataExtractor(context_window=-1)

    def test_zero_context_window(self, sample_chunk: Chunk, sample_document: Document) -> None:
        """Test that zero context window works correctly."""
        extractor = ChunkMetadataExtractor(context_window=0)
        metadata = extractor.extract(sample_chunk, sample_document)

        context = metadata["context"]
        before_without_ellipsis = context["before"].replace("...", "")
        after_without_ellipsis = context["after"].replace("...", "")
        assert len(before_without_ellipsis) == 0
        assert len(after_without_ellipsis) == 0


class TestMetadataSerialization:
    """Tests for metadata serializability."""

    def test_metadata_is_json_serializable(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that extracted metadata is JSON serializable."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        # Should not raise an exception
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["chunk_id"] == metadata["chunk_id"]

    def test_position_info_is_serializable(self, sample_chunk: Chunk) -> None:
        """Test that position info is JSON serializable."""
        position_info = extract_position_info(sample_chunk)

        json_str = json.dumps(position_info)
        parsed = json.loads(json_str)

        assert parsed["start_char"] == position_info["start_char"]

    def test_context_is_serializable(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that context is JSON serializable."""
        context = extract_context(sample_chunk, sample_document)

        json_str = json.dumps(context)
        parsed = json.loads(json_str)

        assert parsed["document_id"] == context["document_id"]

    def test_metadata_with_datetime(
        self, sample_chunk: Chunk, sample_document: Document
    ) -> None:
        """Test that metadata with datetime is serializable."""
        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(sample_chunk, sample_document)

        # datetime should be converted to ISO format string
        assert isinstance(metadata["document_metadata"]["created_at"], str)

        # Should be parseable as datetime
        created_at = datetime.fromisoformat(metadata["document_metadata"]["created_at"])
        assert isinstance(created_at, datetime)

    def test_metadata_with_none_values(self) -> None:
        """Test serialization of metadata with None values."""
        doc = Document(
            id="doc_1",
            content="Test content",
            source="/path/to/doc.txt",
            document_type=None,
            full_metadata=None,
        )
        chunk = Chunk(
            id="doc_1_chunk_0",
            document_id="doc_1",
            content="Test",
            start_idx=0,
            end_idx=4,
            metadata={},
        )

        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(chunk, doc)

        # Should serialize without error
        json_str = json.dumps(metadata)
        parsed = json.loads(json_str)

        assert parsed["document_metadata"]["document_type"] is None
        # author key is not present when full_metadata is None
        assert "author" not in parsed["document_metadata"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_chunk_content(self, sample_document: Document) -> None:
        """Test handling of chunk with empty content."""
        chunk = Chunk(
            id="doc_123_chunk_0",
            document_id="doc_123",
            content="",
            start_idx=0,
            end_idx=0,
            metadata={},
        )

        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(chunk, sample_document)

        assert metadata["content_length"] == 0
        assert metadata["content_hash"] == 0

    def test_single_character_chunk(self, sample_document: Document) -> None:
        """Test handling of single character chunk."""
        chunk = Chunk(
            id="doc_123_chunk_0",
            document_id="doc_123",
            content="T",
            start_idx=0,
            end_idx=1,
            metadata={},
        )

        position_info = extract_position_info(chunk)
        assert position_info["char_count"] == 1

    def test_chunk_larger_than_context_window(self) -> None:
        """Test handling when chunk is larger than context window."""
        doc_content = "A" * 1000  # 1000 characters
        doc = Document(
            id="doc_1",
            content=doc_content,
            source="/path/to/doc.txt",
        )
        chunk = Chunk(
            id="doc_1_chunk_0",
            document_id="doc_1",
            content=doc_content[100:900],  # 800 character chunk
            start_idx=100,
            end_idx=900,
            metadata={},
        )

        extractor = ChunkMetadataExtractor(context_window=50)
        metadata = extractor.extract(chunk, doc)

        # Context should still work, just limited
        assert "before" in metadata["context"]
        assert "after" in metadata["context"]

    def test_document_with_special_characters(self) -> None:
        """Test handling of document with special characters."""
        doc = Document(
            id="doc_1",
            content="Hello\nWorld\tTabbed\"Quotes\"and'apostrophes'中文",
            source="/path/to/doc.txt",
        )
        chunk = Chunk(
            id="doc_1_chunk_0",
            document_id="doc_1",
            content=doc.content,
            start_idx=0,
            end_idx=len(doc.content),
            metadata={},
        )

        extractor = ChunkMetadataExtractor()
        metadata = extractor.extract(chunk, doc)

        # Should serialize without error even with special chars
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)

    def test_relative_position_calculation(self) -> None:
        """Test that relative position is calculated correctly."""
        doc = Document(
            id="doc_1",
            content="A" * 1000,
            source="/path/to/doc.txt",
        )

        # Chunk at start
        chunk_start = Chunk(
            id="doc_1_chunk_0",
            document_id="doc_1",
            content="A" * 100,
            start_idx=0,
            end_idx=100,
            metadata={},
        )

        # Chunk in middle
        chunk_middle = Chunk(
            id="doc_1_chunk_1",
            document_id="doc_1",
            content="A" * 100,
            start_idx=450,
            end_idx=550,
            metadata={},
        )

        # Chunk at end
        chunk_end = Chunk(
            id="doc_1_chunk_2",
            document_id="doc_1",
            content="A" * 100,
            start_idx=900,
            end_idx=1000,
            metadata={},
        )

        extractor = ChunkMetadataExtractor()

        meta_start = extractor.extract(chunk_start, doc)
        meta_middle = extractor.extract(chunk_middle, doc)
        meta_end = extractor.extract(chunk_end, doc)

        # Position should be relative to document length
        assert meta_start["position"]["position"] == 0.0
        assert meta_middle["position"]["position"] == 0.45
        assert meta_end["position"]["position"] == 0.9
