"""Document processing utilities for the GraphRAG system.

This module provides utility functions for document identification,
type detection, and metadata extraction.
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import re

from src.types import DocumentType, DocumentMetadata


# Mapping of file extensions to DocumentType
EXTENSION_TO_DOCUMENT_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MD,
    ".markdown": DocumentType.MD,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".json": DocumentType.JSON,
}


def create_document_id(source: str) -> str:
    """Generate a deterministic document ID from a source path or URL.
    
    This function creates a consistent, reproducible identifier for a document
    based on its source location. The same source will always produce the same ID.
    
    Args:
        source: The source location of the document (file path, URL, etc.).
    
    Returns:
        A deterministic hash-based identifier string prefixed with 'doc-'.
    
    Example:
        ```python
        doc_id = create_document_id("s3://bucket/doc.pdf")
        # Returns: "doc-a1b2c3d4e5f6..."
        
        # Same source always produces same ID
        assert create_document_id("s3://bucket/doc.pdf") == doc_id
        ```
    """
    # Normalize the source string for consistent hashing
    normalized_source = source.strip().lower()
    
    # Create SHA-256 hash of the source
    hash_object = hashlib.sha256(normalized_source.encode("utf-8"))
    hash_hex = hash_object.hexdigest()
    
    # Return with 'doc-' prefix for clarity
    return f"doc-{hash_hex}"


def detect_document_type(
    source: str, 
    content: Optional[str] = None
) -> Optional[DocumentType]:
    """Detect the document type from file extension or content.
    
    This function attempts to determine the document type first by examining
    the file extension from the source path. If the extension is not recognized
    or missing, it can optionally analyze the content for hints.
    
    Args:
        source: The source location of the document (file path, URL, etc.).
        content: Optional document content for content-based detection.
    
    Returns:
        The detected DocumentType, or None if the type cannot be determined.
    
    Example:
        ```python
        doc_type = detect_document_type("report.pdf")
        # Returns: DocumentType.PDF
        
        doc_type = detect_document_type("unknown_file", content="# Heading")
        # Returns: DocumentType.MD (detected from content)
        ```
    """
    # First, try to detect from file extension
    path = Path(source)
    extension = path.suffix.lower()
    
    if extension in EXTENSION_TO_DOCUMENT_TYPE:
        return EXTENSION_TO_DOCUMENT_TYPE[extension]
    
    # If extension not recognized and content is provided, try content-based detection
    if content is not None:
        return _detect_type_from_content(content)
    
    return None


def _detect_type_from_content(content: str) -> Optional[DocumentType]:
    """Detect document type by analyzing content patterns.
    
    Args:
        content: The document content to analyze.
    
    Returns:
        The detected DocumentType, or None if the type cannot be determined.
    """
    content = content.strip()
    
    if not content:
        return None
    
    # Try to detect JSON
    if content.startswith("{") or content.startswith("["):
        try:
            json.loads(content)
            return DocumentType.JSON
        except json.JSONDecodeError:
            pass
    
    # Try to detect HTML
    html_patterns = [
        r"<!DOCTYPE\s+html",
        r"<html",
        r"<head",
        r"<body",
        r"<div",
        r"<p\s*>",
    ]
    content_lower = content.lower()
    for pattern in html_patterns:
        if re.search(pattern, content_lower):
            return DocumentType.HTML
    
    # Try to detect Markdown
    md_patterns = [
        r"^#{1,6}\s+",  # Headings
        r"^\*\s+",  # Unordered lists
        r"^\d+\.\s+",  # Ordered lists
        r"\[.*\]\(.*\)",  # Links
        r"\*\*.*\*\*",  # Bold
        r"\*.*\*",  # Italic
        r"^```",  # Code blocks
        r"^---",  # Horizontal rules
    ]
    md_matches = sum(1 for p in md_patterns if re.search(p, content, re.MULTILINE))
    if md_matches >= 2:
        return DocumentType.MD
    
    # Default to TXT for plain text content
    return DocumentType.TXT


def extract_metadata(
    source: str,
    content: Optional[str] = None,
    file_stats: Optional[os.stat_result] = None,
) -> DocumentMetadata:
    """Extract basic metadata from a document source.
    
    This function extracts available metadata from the document source,
    including file statistics and content analysis. For file-based sources,
    it can extract creation and modification times.
    
    Args:
        source: The source location of the document.
        content: Optional document content for content-based extraction.
        file_stats: Optional os.stat_result for file metadata.
    
    Returns:
        A DocumentMetadata object with extracted information.
    
    Example:
        ```python
        metadata = extract_metadata(
            "/path/to/document.txt",
            content="Document content...",
        )
        print(metadata.title)  # "document"
        ```
    """
    path = Path(source)
    
    # Extract title from filename (without extension)
    title = path.stem if path.stem else None
    
    # Initialize timestamps
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    
    # Extract file timestamps if stats are provided
    if file_stats is not None:
        created = datetime.fromtimestamp(file_stats.st_ctime)
        modified = datetime.fromtimestamp(file_stats.st_mtime)
    
    # Try to extract author from content if provided
    author: Optional[str] = None
    tags: list[str] = []
    
    if content is not None:
        author, content_tags = _extract_metadata_from_content(content)
        if author:
            pass  # Keep extracted author
        if content_tags:
            tags.extend(content_tags)
    
    return DocumentMetadata(
        author=author,
        title=title,
        created=created,
        modified=modified,
        tags=tags,
    )


def _extract_metadata_from_content(content: str) -> tuple[Optional[str], list[str]]:
    """Extract metadata by analyzing document content.
    
    This function looks for common metadata patterns in document content,
    such as YAML front matter in Markdown files or author annotations.
    
    Args:
        content: The document content to analyze.
    
    Returns:
        A tuple of (author, tags) extracted from the content.
    """
    author: Optional[str] = None
    tags: list[str] = []
    
    # Check for YAML front matter (common in Markdown files)
    if content.startswith("---"):
        front_matter_end = content.find("---", 3)
        if front_matter_end != -1:
            front_matter = content[3:front_matter_end]
            
            # Extract author from front matter
            author_match = re.search(
                r"author:\s*['\"]?([^'\"\n]+)['\"]?", 
                front_matter, 
                re.IGNORECASE
            )
            if author_match:
                author = author_match.group(1).strip()
            
            # Extract tags from front matter
            tags_match = re.search(
                r"tags:\s*\[(.*?)\]", 
                front_matter
            )
            if tags_match:
                tag_str = tags_match.group(1)
                tags = [t.strip().strip("'\"") for t in tag_str.split(",")]
            else:
                # Try multi-line tags format
                tags_section = re.search(
                    r"tags:\s*\n((?:\s*-\s*.+\n?)+)", 
                    front_matter
                )
                if tags_section:
                    tag_lines = tags_section.group(1)
                    tags = re.findall(r"-\s*(.+)", tag_lines)
                    tags = [t.strip() for t in tags]
    
    # Check for common author patterns in the first few lines
    if author is None:
        first_lines = content.split("\n")[:10]
        for line in first_lines:
            # Look for "Author: Name" pattern
            author_match = re.search(
                r"author:\s*(.+)", 
                line, 
                re.IGNORECASE
            )
            if author_match:
                author = author_match.group(1).strip()
                break
            
            # Look for "By: Name" pattern
            by_match = re.search(r"^by:\s*(.+)", line, re.IGNORECASE)
            if by_match:
                author = by_match.group(1).strip()
                break
    
    return author, tags
