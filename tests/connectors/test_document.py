"""Tests for document processing utilities.

This module tests the document ID generation, type detection,
and metadata extraction functions.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.connectors.document import (
    create_document_id,
    detect_document_type,
    extract_metadata,
    _detect_type_from_content,
    _extract_metadata_from_content,
)
from src.types import DocumentType, DocumentMetadata


class TestCreateDocumentId:
    """Tests for create_document_id function."""
    
    def test_deterministic_id_generation(self):
        """Test that the same source always produces the same ID."""
        source = "s3://bucket/doc.pdf"
        
        id1 = create_document_id(source)
        id2 = create_document_id(source)
        
        assert id1 == id2, "Same source should produce same ID"
    
    def test_different_sources_produce_different_ids(self):
        """Test that different sources produce different IDs."""
        source1 = "s3://bucket/doc1.pdf"
        source2 = "s3://bucket/doc2.pdf"
        
        id1 = create_document_id(source1)
        id2 = create_document_id(source2)
        
        assert id1 != id2, "Different sources should produce different IDs"
    
    def test_id_has_doc_prefix(self):
        """Test that generated IDs have 'doc-' prefix."""
        source = "path/to/document.txt"
        doc_id = create_document_id(source)
        
        assert doc_id.startswith("doc-"), "ID should start with 'doc-'"
    
    def test_id_is_hash_based(self):
        """Test that ID contains a valid SHA-256 hash."""
        source = "test.pdf"
        doc_id = create_document_id(source)
        
        # Remove prefix and check it's a valid hex string
        hash_part = doc_id[4:]  # Remove 'doc-' prefix
        assert len(hash_part) == 64, "Hash should be 64 characters (SHA-256)"
        assert all(c in "0123456789abcdef" for c in hash_part), "Hash should be hexadecimal"
    
    def test_case_normalization(self):
        """Test that source is normalized for case-insensitive comparison."""
        source1 = "S3://BUCKET/DOC.PDF"
        source2 = "s3://bucket/doc.pdf"
        
        id1 = create_document_id(source1)
        id2 = create_document_id(source2)
        
        assert id1 == id2, "Case should be normalized for consistent IDs"
    
    def test_whitespace_normalization(self):
        """Test that leading/trailing whitespace is normalized."""
        source1 = "  s3://bucket/doc.pdf  "
        source2 = "s3://bucket/doc.pdf"
        
        id1 = create_document_id(source1)
        id2 = create_document_id(source2)
        
        assert id1 == id2, "Whitespace should be normalized"
    
    def test_various_source_formats(self):
        """Test ID generation with various source formats."""
        sources = [
            "s3://bucket/key",
            "/local/path/file.txt",
            "https://example.com/document.pdf",
            "C:\\Windows\\Path\\file.doc",
            "relative/path/file.md",
        ]
        
        ids = [create_document_id(s) for s in sources]
        
        # All IDs should be unique
        assert len(ids) == len(set(ids)), "All sources should produce unique IDs"
        
        # All IDs should have proper format
        for doc_id in ids:
            assert doc_id.startswith("doc-")


class TestDetectDocumentType:
    """Tests for detect_document_type function."""
    
    def test_pdf_detection(self):
        """Test detection of PDF files."""
        assert detect_document_type("document.pdf") == DocumentType.PDF
        assert detect_document_type("report.PDF") == DocumentType.PDF
        assert detect_document_type("/path/to/file.pdf") == DocumentType.PDF
    
    def test_txt_detection(self):
        """Test detection of text files."""
        assert detect_document_type("document.txt") == DocumentType.TXT
        assert detect_document_type("notes.TXT") == DocumentType.TXT
    
    def test_markdown_detection(self):
        """Test detection of Markdown files."""
        assert detect_document_type("README.md") == DocumentType.MD
        assert detect_document_type("GUIDE.markdown") == DocumentType.MD
        assert detect_document_type("doc.MD") == DocumentType.MD
    
    def test_html_detection(self):
        """Test detection of HTML files."""
        assert detect_document_type("index.html") == DocumentType.HTML
        assert detect_document_type("page.htm") == DocumentType.HTML
        assert detect_document_type("web.HTML") == DocumentType.HTML
    
    def test_json_detection(self):
        """Test detection of JSON files."""
        assert detect_document_type("data.json") == DocumentType.JSON
        assert detect_document_type("config.JSON") == DocumentType.JSON
    
    def test_unknown_extension_returns_none(self):
        """Test that unknown extensions return None without content."""
        assert detect_document_type("file.xyz") is None
        assert detect_document_type("file") is None
        assert detect_document_type("file.docx") is None
    
    def test_content_based_json_detection(self):
        """Test JSON detection from content."""
        json_content = '{"key": "value"}'
        assert detect_document_type("unknown", content=json_content) == DocumentType.JSON
        
        json_array = '[1, 2, 3]'
        assert detect_document_type("unknown", content=json_array) == DocumentType.JSON
    
    def test_content_based_html_detection(self):
        """Test HTML detection from content."""
        html_content = "<!DOCTYPE html><html><body>Content</body></html>"
        assert detect_document_type("unknown", content=html_content) == DocumentType.HTML
        
        html_fragment = "<div><p>Some content</p></div>"
        assert detect_document_type("unknown", content=html_fragment) == DocumentType.HTML
    
    def test_content_based_markdown_detection(self):
        """Test Markdown detection from content."""
        md_content = """# Heading
        
This is a paragraph with **bold** text.

- List item 1
- List item 2

[Link](https://example.com)
"""
        assert detect_document_type("unknown", content=md_content) == DocumentType.MD
    
    def test_content_based_text_fallback(self):
        """Test that plain text content defaults to TXT."""
        plain_text = "This is just plain text without any special formatting."
        assert detect_document_type("unknown", content=plain_text) == DocumentType.TXT
    
    def test_extension_takes_precedence_over_content(self):
        """Test that file extension is used even if content suggests different type."""
        # A .txt file with JSON content should still be detected as TXT
        json_content = '{"key": "value"}'
        assert detect_document_type("file.txt", content=json_content) == DocumentType.TXT


class TestDetectTypeFromContent:
    """Tests for _detect_type_from_content helper function."""
    
    def test_empty_content_returns_none(self):
        """Test that empty content returns None."""
        assert _detect_type_from_content("") is None
        assert _detect_type_from_content("   ") is None
    
    def test_json_object_detection(self):
        """Test JSON object detection."""
        content = '{"name": "test", "value": 123}'
        assert _detect_type_from_content(content) == DocumentType.JSON
    
    def test_json_array_detection(self):
        """Test JSON array detection."""
        content = '[{"id": 1}, {"id": 2}]'
        assert _detect_type_from_content(content) == DocumentType.JSON
    
    def test_invalid_json_not_detected_as_json(self):
        """Test that invalid JSON is not detected as JSON."""
        content = '{not valid json}'
        result = _detect_type_from_content(content)
        # Should fall back to TXT or None
        assert result in (DocumentType.TXT, None) or result != DocumentType.JSON
    
    def test_html_with_doctype(self):
        """Test HTML detection with DOCTYPE."""
        content = "<!DOCTYPE html><html><head></head><body></body></html>"
        assert _detect_type_from_content(content) == DocumentType.HTML
    
    def test_html_fragment(self):
        """Test HTML fragment detection."""
        content = "<div class='container'><p>Text</p></div>"
        assert _detect_type_from_content(content) == DocumentType.HTML
    
    def test_markdown_with_headings(self):
        """Test Markdown detection with headings."""
        content = "# Main Heading\n\n## Subheading\n\nParagraph text with **bold**."
        assert _detect_type_from_content(content) == DocumentType.MD
    
    def test_markdown_with_links_and_formatting(self):
        """Test Markdown detection with links and formatting."""
        content = "Check out [this link](https://example.com) for **more info**."
        assert _detect_type_from_content(content) == DocumentType.MD


class TestExtractMetadata:
    """Tests for extract_metadata function."""
    
    def test_title_extraction_from_filename(self):
        """Test that title is extracted from filename."""
        metadata = extract_metadata("/path/to/my_document.pdf")
        assert metadata.title == "my_document"
    
    def test_title_extraction_from_simple_filename(self):
        """Test title extraction from simple filename."""
        metadata = extract_metadata("report.txt")
        assert metadata.title == "report"
    
    def test_title_extraction_from_url(self):
        """Test title extraction from URL-like source."""
        metadata = extract_metadata("s3://bucket/documents/report.pdf")
        assert metadata.title == "report"
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        metadata = extract_metadata("document.txt")
        
        assert metadata.author is None
        assert metadata.created is None
        assert metadata.modified is None
        assert metadata.tags == []
    
    def test_file_stats_extraction(self):
        """Test extraction of timestamps from file stats."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            temp_path = f.name
        
        try:
            stats = os.stat(temp_path)
            metadata = extract_metadata(temp_path, file_stats=stats)
            
            assert metadata.created is not None
            assert metadata.modified is not None
            assert isinstance(metadata.created, datetime)
            assert isinstance(metadata.modified, datetime)
        finally:
            os.unlink(temp_path)
    
    def test_author_extraction_from_front_matter(self):
        """Test author extraction from YAML front matter."""
        content = """---
author: John Doe
title: Test Document
---
# Content starts here
"""
        metadata = extract_metadata("test.md", content=content)
        assert metadata.author == "John Doe"
    
    def test_author_extraction_quoted(self):
        """Test author extraction with quotes in front matter."""
        content = """---
author: "Jane Smith"
---
Content
"""
        metadata = extract_metadata("test.md", content=content)
        assert metadata.author == "Jane Smith"
    
    def test_tags_extraction_from_front_matter_array(self):
        """Test tags extraction from YAML array format."""
        content = """---
tags: [python, testing, documentation]
---
Content
"""
        metadata = extract_metadata("test.md", content=content)
        assert "python" in metadata.tags
        assert "testing" in metadata.tags
        assert "documentation" in metadata.tags
    
    def test_tags_extraction_from_front_matter_list(self):
        """Test tags extraction from YAML list format."""
        content = """---
tags:
  - python
  - testing
  - documentation
---
Content
"""
        metadata = extract_metadata("test.md", content=content)
        assert "python" in metadata.tags
        assert "testing" in metadata.tags
        assert "documentation" in metadata.tags
    
    def test_author_extraction_from_content_pattern(self):
        """Test author extraction from 'Author:' pattern in content."""
        content = """First line of document
Author: Alice Johnson
More content here
"""
        metadata = extract_metadata("document.txt", content=content)
        assert metadata.author == "Alice Johnson"
    
    def test_author_extraction_by_pattern(self):
        """Test author extraction from 'By:' pattern."""
        content = """Document Header
By: Bob Writer

The document content...
"""
        metadata = extract_metadata("article.txt", content=content)
        assert metadata.author == "Bob Writer"
    
    def test_combined_metadata_extraction(self):
        """Test combined extraction from source and content."""
        content = """---
author: Test Author
tags: [tag1, tag2]
---
# Document Title

Content here.
"""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(content.encode())
            temp_path = f.name
        
        try:
            stats = os.stat(temp_path)
            metadata = extract_metadata(temp_path, content=content, file_stats=stats)
            
            # Title from filename
            assert metadata.title is not None
            # Author from front matter
            assert metadata.author == "Test Author"
            # Tags from front matter
            assert "tag1" in metadata.tags
            assert "tag2" in metadata.tags
            # Timestamps from file stats
            assert metadata.created is not None
            assert metadata.modified is not None
        finally:
            os.unlink(temp_path)


class TestExtractMetadataFromContent:
    """Tests for _extract_metadata_from_content helper function."""
    
    def test_no_metadata_in_plain_text(self):
        """Test that plain text returns no author or tags."""
        author, tags = _extract_metadata_from_content("Just plain text content.")
        assert author is None
        assert tags == []
    
    def test_front_matter_author_extraction(self):
        """Test author extraction from YAML front matter."""
        content = """---
author: Test Author
---
Content
"""
        author, tags = _extract_metadata_from_content(content)
        assert author == "Test Author"
    
    def test_front_matter_tags_array(self):
        """Test tags extraction from YAML array in front matter."""
        content = """---
tags: [one, two, three]
---
Content
"""
        author, tags = _extract_metadata_from_content(content)
        assert tags == ["one", "two", "three"]
    
    def test_front_matter_tags_list(self):
        """Test tags extraction from YAML list in front matter."""
        content = """---
tags:
  - one
  - two
  - three
---
Content
"""
        author, tags = _extract_metadata_from_content(content)
        assert "one" in tags
        assert "two" in tags
        assert "three" in tags
    
    def test_author_line_pattern(self):
        """Test author extraction from 'Author:' line pattern."""
        content = "Some text\nAuthor: John Doe\nMore text"
        author, tags = _extract_metadata_from_content(content)
        assert author == "John Doe"
    
    def test_by_line_pattern(self):
        """Test author extraction from 'By:' line pattern."""
        content = "Article Title\nBy: Jane Smith\n\nContent"
        author, tags = _extract_metadata_from_content(content)
        assert author == "Jane Smith"


class TestDocumentTypeIntegration:
    """Integration tests for DocumentType with Document model."""
    
    def test_document_with_document_type(self):
        """Test creating a Document with document_type."""
        from src.types import Document
        
        doc = Document(
            id="doc-001",
            content="Test content",
            source="test.pdf",
            document_type=DocumentType.PDF,
        )
        
        assert doc.document_type == DocumentType.PDF
    
    def test_document_with_full_metadata(self):
        """Test creating a Document with full_metadata."""
        from src.types import Document
        
        full_meta = DocumentMetadata(
            author="Test Author",
            title="Test Title",
            tags=["test", "example"],
        )
        
        doc = Document(
            id="doc-002",
            content="Test content",
            source="test.md",
            document_type=DocumentType.MD,
            full_metadata=full_meta,
        )
        
        assert doc.full_metadata is not None
        assert doc.full_metadata.author == "Test Author"
        assert doc.full_metadata.title == "Test Title"
        assert "test" in doc.full_metadata.tags
    
    def test_document_type_enum_values(self):
        """Test DocumentType enum values."""
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.TXT.value == "txt"
        assert DocumentType.MD.value == "md"
        assert DocumentType.HTML.value == "html"
        assert DocumentType.JSON.value == "json"
    
    def test_document_metadata_model_defaults(self):
        """Test DocumentMetadata default values."""
        metadata = DocumentMetadata()
        
        assert metadata.author is None
        assert metadata.title is None
        assert metadata.created is None
        assert metadata.modified is None
        assert metadata.tags == []
