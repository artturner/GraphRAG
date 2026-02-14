"""Tests for the LocalConnector class.

This module contains comprehensive tests for the LocalConnector,
testing document loading from the local filesystem.
"""

import logging
from pathlib import Path
from typing import Generator

import pytest
from pypdf import PdfWriter

from src.connectors.local import LocalConnector, SUPPORTED_EXTENSIONS
from src.exceptions import ConnectorError
from src.types import DocumentType


# Fixtures


@pytest.fixture
def temp_documents_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample documents.
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory.
    
    Returns:
        Path to the temporary documents directory.
    """
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    
    # Create a sample text file
    txt_file = docs_dir / "test.txt"
    txt_file.write_text("This is a test text file.\nIt has multiple lines.", encoding="utf-8")
    
    # Create a sample markdown file
    md_file = docs_dir / "test.md"
    md_file.write_text(
        "# Test Markdown\n\nThis is a test markdown file.\n\n## Features\n- Item 1\n- Item 2",
        encoding="utf-8"
    )
    
    # Create a sample HTML file
    html_file = docs_dir / "test.html"
    html_file.write_text(
        "<!DOCTYPE html><html><body><h1>Test HTML</h1><p>Content here.</p></body></html>",
        encoding="utf-8"
    )
    
    # Create a sample PDF file
    pdf_file = docs_dir / "test.pdf"
    writer = PdfWriter()
    # Add a blank page with text
    page = writer.add_blank_page(width=200, height=200)
    # Note: pypdf doesn't easily add text to pages, so we create a simple PDF
    with open(pdf_file, "wb") as f:
        writer.write(f)
    
    # Create an unsupported file type
    unsupported_file = docs_dir / "test.xyz"
    unsupported_file.write_text("This file type is not supported.", encoding="utf-8")
    
    # Create a subdirectory with files
    subdir = docs_dir / "subdir"
    subdir.mkdir()
    sub_txt = subdir / "nested.txt"
    sub_txt.write_text("Nested text file content.", encoding="utf-8")
    
    return docs_dir


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the path to the fixtures directory.
    
    Returns:
        Path to the tests/fixtures/documents directory.
    """
    return Path(__file__).parent.parent / "fixtures" / "documents"


@pytest.fixture
def connector(temp_documents_dir: Path) -> LocalConnector:
    """Create a LocalConnector instance for testing.
    
    Args:
        temp_documents_dir: Fixture providing a temporary documents directory.
    
    Returns:
        LocalConnector instance pointing to the temp directory.
    """
    return LocalConnector(temp_documents_dir)


# Test Classes


class TestLocalConnectorInit:
    """Tests for LocalConnector initialization."""
    
    def test_init_with_string_path(self, temp_documents_dir: Path) -> None:
        """Test initialization with a string path."""
        connector = LocalConnector(str(temp_documents_dir))
        assert connector.source == str(temp_documents_dir)
        assert connector.source_path == temp_documents_dir
    
    def test_init_with_path_object(self, temp_documents_dir: Path) -> None:
        """Test initialization with a Path object."""
        connector = LocalConnector(temp_documents_dir)
        assert connector.source == str(temp_documents_dir)
        assert connector.source_path == temp_documents_dir


class TestValidateSource:
    """Tests for the validate_source method."""
    
    def test_validate_source_existing_directory(
        self, connector: LocalConnector
    ) -> None:
        """Test validate_source returns True for existing directory."""
        assert connector.validate_source() is True
    
    def test_validate_source_missing_directory(self, tmp_path: Path) -> None:
        """Test validate_source returns False for missing directory."""
        missing_dir = tmp_path / "nonexistent"
        connector = LocalConnector(missing_dir)
        assert connector.validate_source() is False
    
    def test_validate_source_file_not_directory(
        self, tmp_path: Path
    ) -> None:
        """Test validate_source returns False for a file path."""
        # Create a file (not a directory)
        file_path = tmp_path / "file.txt"
        file_path.write_text("content", encoding="utf-8")
        
        connector = LocalConnector(file_path)
        assert connector.validate_source() is False


class TestListDocuments:
    """Tests for the list_documents method."""
    
    def test_list_documents_returns_supported_files(
        self, connector: LocalConnector
    ) -> None:
        """Test that list_documents returns only supported file types."""
        doc_list = connector.list_documents()
        
        # Check that all returned paths have supported extensions
        for doc_path in doc_list:
            path = Path(doc_path)
            assert path.suffix.lower() in SUPPORTED_EXTENSIONS
    
    def test_list_documents_includes_nested_files(
        self, connector: LocalConnector
    ) -> None:
        """Test that list_documents includes files in subdirectories."""
        doc_list = connector.list_documents()
        
        # Check that nested file is included
        assert "subdir/nested.txt" in doc_list or "subdir\\nested.txt" in doc_list
    
    def test_list_documents_excludes_unsupported(
        self, connector: LocalConnector
    ) -> None:
        """Test that list_documents excludes unsupported file types."""
        doc_list = connector.list_documents()
        
        # Check that .xyz file is not included
        for doc_path in doc_list:
            assert not doc_path.endswith(".xyz")
    
    def test_list_documents_raises_for_missing_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that list_documents raises ConnectorError for missing directory."""
        missing_dir = tmp_path / "nonexistent"
        connector = LocalConnector(missing_dir)
        
        with pytest.raises(ConnectorError) as exc_info:
            connector.list_documents()
        
        assert "does not exist" in str(exc_info.value)
    
    def test_list_documents_returns_relative_paths(
        self, connector: LocalConnector, temp_documents_dir: Path
    ) -> None:
        """Test that list_documents returns paths relative to source directory."""
        doc_list = connector.list_documents()
        
        for doc_path in doc_list:
            # The path should not be absolute
            assert not Path(doc_path).is_absolute()
            # The full path should exist
            full_path = temp_documents_dir / doc_path
            assert full_path.exists()


class TestLoad:
    """Tests for the load method."""
    
    def test_load_returns_documents(self, connector: LocalConnector) -> None:
        """Test that load returns a list of Document objects."""
        documents = connector.load()
        
        assert isinstance(documents, list)
        assert len(documents) > 0
    
    def test_load_text_file(self, connector: LocalConnector) -> None:
        """Test loading a text file."""
        documents = connector.load()
        
        # Find the text document (test.txt specifically)
        txt_docs = [d for d in documents if d.source.endswith("test.txt")]
        assert len(txt_docs) > 0
        
        txt_doc = txt_docs[0]
        assert "test text file" in txt_doc.content
        assert txt_doc.document_type == DocumentType.TXT
    
    def test_load_markdown_file(self, connector: LocalConnector) -> None:
        """Test loading a markdown file."""
        documents = connector.load()
        
        # Find the markdown document
        md_docs = [d for d in documents if d.source.endswith(".md")]
        assert len(md_docs) > 0
        
        md_doc = md_docs[0]
        assert "Test Markdown" in md_doc.content
        assert md_doc.document_type == DocumentType.MD
    
    def test_load_html_file(self, connector: LocalConnector) -> None:
        """Test loading an HTML file."""
        documents = connector.load()
        
        # Find the HTML document
        html_docs = [d for d in documents if d.source.endswith(".html")]
        assert len(html_docs) > 0
        
        html_doc = html_docs[0]
        assert "Test HTML" in html_doc.content
        assert html_doc.document_type == DocumentType.HTML
    
    def test_load_pdf_file(self, connector: LocalConnector) -> None:
        """Test loading a PDF file."""
        documents = connector.load()
        
        # Find the PDF document
        pdf_docs = [d for d in documents if d.source.endswith(".pdf")]
        assert len(pdf_docs) > 0
        
        pdf_doc = pdf_docs[0]
        assert pdf_doc.document_type == DocumentType.PDF
        # Note: Our test PDF is blank, so content may be empty
    
    def test_load_skips_unsupported_files(
        self, connector: LocalConnector, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that load skips unsupported file types with a warning."""
        with caplog.at_level(logging.WARNING):
            documents = connector.load()
        
        # Check that no .xyz files were loaded
        for doc in documents:
            assert not doc.source.endswith(".xyz")
    
    def test_load_raises_for_missing_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that load raises ConnectorError for missing directory."""
        missing_dir = tmp_path / "nonexistent"
        connector = LocalConnector(missing_dir)
        
        with pytest.raises(ConnectorError) as exc_info:
            connector.load()
        
        assert "does not exist" in str(exc_info.value)
    
    def test_load_document_has_metadata(
        self, connector: LocalConnector
    ) -> None:
        """Test that loaded documents have proper metadata."""
        documents = connector.load()
        
        for doc in documents:
            assert doc.id is not None
            assert doc.id.startswith("doc-")
            assert doc.source is not None
            assert doc.content is not None
            assert doc.document_type is not None
            assert doc.metadata is not None
            assert "absolute_path" in doc.metadata
            assert "file_size" in doc.metadata
    
    def test_load_document_has_full_metadata(
        self, connector: LocalConnector
    ) -> None:
        """Test that loaded documents have full_metadata with title."""
        documents = connector.load()
        
        for doc in documents:
            assert doc.full_metadata is not None
            assert doc.full_metadata.title is not None


class TestLoadFixtureDocuments:
    """Tests for loading documents from the fixtures directory."""
    
    def test_load_sample_txt(self, fixtures_dir: Path) -> None:
        """Test loading the sample.txt fixture file."""
        if not fixtures_dir.exists():
            pytest.skip("Fixtures directory not found")
        
        connector = LocalConnector(fixtures_dir)
        documents = connector.load()
        
        txt_docs = [d for d in documents if d.source == "sample.txt"]
        assert len(txt_docs) == 1
        
        txt_doc = txt_docs[0]
        assert "sample text document" in txt_doc.content.lower()
        assert txt_doc.document_type == DocumentType.TXT
    
    def test_load_sample_md(self, fixtures_dir: Path) -> None:
        """Test loading the sample.md fixture file."""
        if not fixtures_dir.exists():
            pytest.skip("Fixtures directory not found")
        
        connector = LocalConnector(fixtures_dir)
        documents = connector.load()
        
        md_docs = [d for d in documents if d.source == "sample.md"]
        assert len(md_docs) == 1
        
        md_doc = md_docs[0]
        assert "Sample Markdown" in md_doc.content
        assert md_doc.document_type == DocumentType.MD
    
    def test_load_sample_html(self, fixtures_dir: Path) -> None:
        """Test loading the sample.html fixture file."""
        if not fixtures_dir.exists():
            pytest.skip("Fixtures directory not found")
        
        connector = LocalConnector(fixtures_dir)
        documents = connector.load()
        
        html_docs = [d for d in documents if d.source == "sample.html"]
        assert len(html_docs) == 1
        
        html_doc = html_docs[0]
        assert "Sample HTML" in html_doc.content
        assert html_doc.document_type == DocumentType.HTML


class TestEncodingDetection:
    """Tests for encoding detection in text files."""
    
    def test_load_utf8_file(self, temp_documents_dir: Path) -> None:
        """Test loading a UTF-8 encoded file."""
        # Create a file with UTF-8 content
        utf8_file = temp_documents_dir / "utf8.txt"
        utf8_file.write_text("Hello, world! \u4e16\u754c", encoding="utf-8")
        
        connector = LocalConnector(temp_documents_dir)
        documents = connector.load()
        
        utf8_docs = [d for d in documents if d.source.endswith("utf8.txt")]
        assert len(utf8_docs) == 1
        assert "\u4e16\u754c" in utf8_docs[0].content
    
    def test_load_latin1_file(self, temp_documents_dir: Path) -> None:
        """Test loading a Latin-1 encoded file."""
        # Create a file with Latin-1 content
        latin1_file = temp_documents_dir / "latin1.txt"
        # Write bytes directly to avoid encoding issues
        latin1_file.write_bytes("Caf\xe9".encode("latin-1"))
        
        connector = LocalConnector(temp_documents_dir)
        documents = connector.load()
        
        latin1_docs = [d for d in documents if d.source.endswith("latin1.txt")]
        assert len(latin1_docs) == 1
        # The content should be decoded (may have replacement characters)
        assert len(latin1_docs[0].content) > 0


class TestSupportedExtensions:
    """Tests for supported file extensions."""
    
    def test_supported_extensions_include_required_types(self) -> None:
        """Test that SUPPORTED_EXTENSIONS includes required file types."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".html" in SUPPORTED_EXTENSIONS
        assert ".htm" in SUPPORTED_EXTENSIONS


class TestDocumentIdGeneration:
    """Tests for document ID generation."""
    
    def test_same_source_produces_same_id(
        self, connector: LocalConnector
    ) -> None:
        """Test that the same source path produces the same document ID."""
        documents1 = connector.load()
        documents2 = connector.load()
        
        # Sort by source for comparison
        docs1_sorted = sorted(documents1, key=lambda d: d.source)
        docs2_sorted = sorted(documents2, key=lambda d: d.source)
        
        for doc1, doc2 in zip(docs1_sorted, docs2_sorted):
            assert doc1.id == doc2.id
    
    def test_different_sources_produce_different_ids(
        self, connector: LocalConnector
    ) -> None:
        """Test that different source paths produce different document IDs."""
        documents = connector.load()
        
        ids = [doc.id for doc in documents]
        assert len(ids) == len(set(ids))  # All IDs should be unique