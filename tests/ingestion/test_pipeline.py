"""Tests for the ingestion pipeline.

This module contains comprehensive tests for the IngestionPipeline class
and its associated components.
"""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.connectors.base import BaseConnector
from src.connectors.local import LocalConnector
from src.ingestion.chunking import FixedSizeChunker, SentenceChunker
from src.ingestion.cleaning import TextCleaner
from src.ingestion.pipeline import IngestProgress, IngestionPipeline
from src.types import Chunk, Document, DocumentType, IngestResult


class TestIngestProgress:
    """Tests for IngestProgress dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        progress = IngestProgress()
        
        assert progress.documents_processed == 0
        assert progress.chunks_created == 0
        assert progress.current_file is None
        assert progress.errors == []
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        progress = IngestProgress(
            documents_processed=5,
            chunks_created=42,
            current_file="test.txt",
            errors=["error1"],
        )
        
        assert progress.documents_processed == 5
        assert progress.chunks_created == 42
        assert progress.current_file == "test.txt"
        assert progress.errors == ["error1"]
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        progress = IngestProgress(
            documents_processed=3,
            chunks_created=15,
            current_file="doc.txt",
            errors=["err"],
        )
        
        result = progress.to_dict()
        
        assert result["documents_processed"] == 3
        assert result["chunks_created"] == 15
        assert result["current_file"] == "doc.txt"
        assert result["errors"] == ["err"]


class TestIngestionPipelineInit:
    """Tests for IngestionPipeline initialization."""
    
    def test_init_with_components(self):
        """Test initialization with all components."""
        connector = MagicMock(spec=BaseConnector)
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        assert pipeline.connector is connector
        assert pipeline.cleaner is cleaner
        assert pipeline.chunker is chunker
    
    def test_repr(self):
        """Test string representation of pipeline."""
        connector = LocalConnector("./data")
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        repr_str = repr(pipeline)
        
        assert "IngestionPipeline" in repr_str
        assert "LocalConnector" in repr_str
        assert "TextCleaner" in repr_str
        assert "FixedSizeChunker" in repr_str


class TestIngestionPipelineRun:
    """Tests for the run method."""
    
    def test_full_pipeline_with_local_connector(self, tmp_path: Path):
        """Test full pipeline with local connector."""
        # Create test documents
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        
        doc1 = doc_dir / "doc1.txt"
        doc1.write_text("This is the first test document. " * 20)
        
        doc2 = doc_dir / "doc2.txt"
        doc2.write_text("This is the second test document. " * 20)
        
        # Create pipeline
        connector = LocalConnector(str(doc_dir))
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=200)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        # Run pipeline
        result = pipeline.run()
        
        assert result.documents_count == 2
        assert result.chunks_count > 0
        assert result.errors == []
    
    def test_documents_are_cleaned_before_chunking(self):
        """Test that documents are cleaned before chunking."""
        # Create mock connector with HTML content
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id="doc-1",
                content="<p>  Hello   <b>World</b>  </p>",
                source="test.html",
                document_type=DocumentType.HTML,
            )
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        
        # Get the chunks created
        assert result.documents_count == 1
        assert result.chunks_count == 1
        
        # Verify content was cleaned (HTML removed, whitespace normalized)
        chunks = pipeline.process_document(connector.load.return_value[0])
        assert "<p>" not in chunks[0].content
        assert "<b>" not in chunks[0].content
        assert "Hello World" in chunks[0].content
    
    def test_chunks_have_correct_metadata(self):
        """Test that chunks have correct metadata."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id="test-doc-123",
                content="This is test content for chunking. " * 10,
                source="/path/to/document.txt",
                document_type=DocumentType.TXT,
                metadata={"category": "test", "author": "tester"},
            )
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=100)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        
        assert result.documents_count == 1
        assert result.chunks_count > 0
        
        # Check chunk metadata
        chunks = pipeline.process_document(connector.load.return_value[0])
        for chunk in chunks:
            assert chunk.document_id == "test-doc-123"
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "/path/to/document.txt"
            assert "chunk_meta" in chunk.metadata
            assert "chunk_index" in chunk.metadata["chunk_meta"]
            assert "total_chunks" in chunk.metadata["chunk_meta"]
    
    def test_error_handling_for_corrupted_files(self, tmp_path: Path):
        """Test error handling for corrupted files."""
        # Create a file that will cause an error during processing
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        
        # Create a valid file
        valid_doc = doc_dir / "valid.txt"
        valid_doc.write_text("This is a valid document.")
        
        # Create pipeline with mock that raises an error
        connector = LocalConnector(str(doc_dir))
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        # Patch _process_document to simulate error
        original_process = pipeline._process_document
        
        def mock_process(doc):
            if "valid" in doc.source:
                raise ValueError("Simulated processing error")
            return original_process(doc)
        
        with patch.object(pipeline, '_process_document', side_effect=mock_process):
            result = pipeline.run()
        
        assert result.documents_count == 0
        assert len(result.errors) > 0
        assert "Simulated processing error" in result.errors[0]
    
    def test_error_handling_for_connector_failure(self):
        """Test error handling when connector fails to load."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.side_effect = RuntimeError("Failed to connect to source")
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        
        assert result.documents_count == 0
        assert result.chunks_count == 0
        assert len(result.errors) > 0
        assert "Failed to connect to source" in result.errors[0]
    
    def test_empty_documents_list(self):
        """Test pipeline with no documents."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = []
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        
        assert result.documents_count == 0
        assert result.chunks_count == 0
        assert result.errors == []


class TestIngestionPipelineRunWithProgress:
    """Tests for the run_with_progress method."""
    
    def test_progress_reporting(self):
        """Test that progress is reported correctly."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id=f"doc-{i}",
                content=f"Document {i} content for testing. " * 10,
                source=f"/path/doc{i}.txt",
                document_type=DocumentType.TXT,
            )
            for i in range(3)
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=200)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        # Collect progress snapshots (the same object is yielded each time)
        progress_snapshots = []
        for progress in pipeline.run_with_progress():
            # Take a snapshot of the current state
            progress_snapshots.append(IngestProgress(
                documents_processed=progress.documents_processed,
                chunks_created=progress.chunks_created,
                current_file=progress.current_file,
                errors=progress.errors.copy(),
            ))
        
        assert len(progress_snapshots) == 3
        
        # Check first progress
        assert progress_snapshots[0].documents_processed == 1
        assert progress_snapshots[0].current_file == "/path/doc0.txt"
        
        # Check last progress
        assert progress_snapshots[-1].documents_processed == 3
        assert progress_snapshots[-1].chunks_created > 0
    
    def test_progress_with_errors(self):
        """Test progress reporting with errors."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id="doc-1",
                content="Valid content",
                source="/path/doc1.txt",
                document_type=DocumentType.TXT,
            ),
            Document(
                id="doc-2",
                content="Error content",
                source="/path/doc2.txt",
                document_type=DocumentType.TXT,
            ),
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        # Patch to simulate error on second document
        original_process = pipeline._process_document
        call_count = [0]
        
        def mock_process(doc):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Processing error")
            return original_process(doc)
        
        # Collect progress snapshots
        progress_snapshots = []
        with patch.object(pipeline, '_process_document', side_effect=mock_process):
            for progress in pipeline.run_with_progress():
                progress_snapshots.append(IngestProgress(
                    documents_processed=progress.documents_processed,
                    chunks_created=progress.chunks_created,
                    current_file=progress.current_file,
                    errors=progress.errors.copy(),
                ))
        
        assert len(progress_snapshots) == 2
        assert progress_snapshots[0].documents_processed == 1
        assert progress_snapshots[0].errors == []
        assert progress_snapshots[1].documents_processed == 1  # Still 1 because second failed
        assert len(progress_snapshots[1].errors) == 1
    
    def test_progress_is_iterator(self):
        """Test that run_with_progress returns an iterator."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id="doc-1",
                content="Content",
                source="test.txt",
            )
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run_with_progress()
        
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')
    
    def test_progress_with_connector_error(self):
        """Test progress when connector fails."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.side_effect = RuntimeError("Connection failed")
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        progress_list = list(pipeline.run_with_progress())
        
        assert len(progress_list) == 1
        assert progress_list[0].documents_processed == 0
        assert progress_list[0].chunks_created == 0
        assert len(progress_list[0].errors) == 1
        assert "Connection failed" in progress_list[0].errors[0]


class TestIngestionPipelineProcessDocument:
    """Tests for the process_document method."""
    
    def test_process_single_document(self):
        """Test processing a single document."""
        connector = MagicMock(spec=BaseConnector)
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=100)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        doc = Document(
            id="test-doc",
            content="This is a test document. " * 10,
            source="test.txt",
            document_type=DocumentType.TXT,
        )
        
        chunks = pipeline.process_document(doc)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.document_id == "test-doc" for chunk in chunks)
    
    def test_process_document_with_html(self):
        """Test processing document with HTML content."""
        connector = MagicMock(spec=BaseConnector)
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        doc = Document(
            id="html-doc",
            content="<html><body><p>Hello World</p></body></html>",
            source="test.html",
            document_type=DocumentType.HTML,
        )
        
        chunks = pipeline.process_document(doc)
        
        assert len(chunks) == 1
        assert "<html>" not in chunks[0].content
        assert "<p>" not in chunks[0].content
        assert "Hello World" in chunks[0].content
    
    def test_process_document_with_sentence_chunker(self):
        """Test processing document with sentence chunker."""
        connector = MagicMock(spec=BaseConnector)
        cleaner = TextCleaner()
        chunker = SentenceChunker(min_size=50, max_size=200)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        doc = Document(
            id="sentence-doc",
            content="First sentence here. Second sentence follows. Third sentence added. "
                    "Fourth sentence continues. Fifth sentence ends the paragraph.",
            source="test.txt",
            document_type=DocumentType.TXT,
        )
        
        chunks = pipeline.process_document(doc)
        
        assert len(chunks) > 0
        assert all(chunk.document_id == "sentence-doc" for chunk in chunks)
        # Check that chunk metadata includes sentence info
        for chunk in chunks:
            assert "sentence_count" in chunk.metadata


class TestIngestionPipelineIntegration:
    """Integration tests for the ingestion pipeline."""
    
    def test_full_pipeline_with_fixture_documents(self):
        """Test full pipeline with fixture documents."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "documents"
        
        if not fixture_path.exists():
            pytest.skip("Fixture directory not found")
        
        connector = LocalConnector(str(fixture_path))
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=300)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        
        # Should have processed at least the sample documents
        assert result.documents_count >= 1
        assert result.chunks_count >= 1
        assert result.errors == []
    
    def test_pipeline_with_different_chunkers(self, tmp_path: Path):
        """Test pipeline with different chunker types."""
        # Create test document
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        
        doc = doc_dir / "test.txt"
        doc.write_text("First sentence. Second sentence. Third sentence. " * 20)
        
        connector = LocalConnector(str(doc_dir))
        cleaner = TextCleaner()
        
        # Test with FixedSizeChunker
        pipeline_fixed = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=FixedSizeChunker(chunk_size=200),
        )
        result_fixed = pipeline_fixed.run()
        
        # Test with SentenceChunker
        pipeline_sentence = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=SentenceChunker(min_size=100, max_size=300),
        )
        result_sentence = pipeline_sentence.run()
        
        # Both should succeed
        assert result_fixed.documents_count == 1
        assert result_fixed.chunks_count > 0
        assert result_sentence.documents_count == 1
        assert result_sentence.chunks_count > 0
        
        # Different chunkers may produce different numbers of chunks
        # (not necessarily different, but both should work)
    
    def test_pipeline_preserves_document_metadata(self):
        """Test that pipeline preserves document metadata in chunks."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id="meta-doc",
                content="Content for metadata test. " * 10,
                source="/path/to/meta.txt",
                document_type=DocumentType.TXT,
                metadata={
                    "author": "Test Author",
                    "category": "Test Category",
                    "custom_field": "custom_value",
                },
            )
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=200)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        chunks = pipeline.process_document(connector.load.return_value[0])
        
        # Check that metadata is preserved
        for chunk in chunks:
            assert chunk.metadata.get("author") == "Test Author"
            assert chunk.metadata.get("category") == "Test Category"
            assert chunk.metadata.get("custom_field") == "custom_value"
    
    def test_pipeline_with_empty_content(self):
        """Test pipeline with document that becomes empty after cleaning."""
        connector = MagicMock(spec=BaseConnector)
        connector.load.return_value = [
            Document(
                id="empty-doc",
                content="",  # Empty content
                source="empty.txt",
                document_type=DocumentType.TXT,
            )
        ]
        
        cleaner = TextCleaner()
        chunker = FixedSizeChunker(chunk_size=500)
        
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        
        result = pipeline.run()
        
        # Document should be processed but produce no chunks
        assert result.documents_count == 1
        assert result.chunks_count == 0
