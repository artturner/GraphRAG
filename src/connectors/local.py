"""Local filesystem connector for loading documents.

This module provides a connector for loading documents from the local filesystem,
supporting various file formats including PDF, TXT, MD, and HTML.
"""

import logging
import os
from pathlib import Path
from typing import Union

from src.connectors.base import BaseConnector
from src.connectors.document import (
    create_document_id,
    detect_document_type,
    extract_metadata,
)
from src.exceptions import ConnectorError
from src.types import Document, DocumentType

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".html", ".htm"}


class LocalConnector(BaseConnector):
    """Connector for loading documents from the local filesystem.
    
    This connector scans a directory for supported document files and loads
    them into Document objects. It supports PDF, TXT, MD, and HTML formats.
    
    Attributes:
        source_path: Path to the directory containing documents.
    
    Example:
        ```python
        from src.connectors.local import LocalConnector
        
        connector = LocalConnector(source_path="./data/documents")
        documents = connector.load()
        for doc in documents:
            print(f"Loaded: {doc.source} ({len(doc.content)} chars)")
        ```
    """
    
    def __init__(self, source_path: Union[str, Path]) -> None:
        """Initialize the local connector with a source directory.
        
        Args:
            source_path: Path to the directory containing documents.
                Can be a string or Path object.
        
        Raises:
            ConnectorError: If the source path is not a valid directory.
        """
        self._source_path = Path(source_path)
        # Call parent constructor with string representation
        super().__init__(str(self._source_path))
    
    @property
    def source_path(self) -> Path:
        """Return the source path as a Path object.
        
        Returns:
            The source directory as a Path object.
        """
        return self._source_path
    
    def validate_source(self) -> bool:
        """Check if the source directory exists and is accessible.
        
        Returns:
            True if the source directory exists and is accessible.
        
        Example:
            ```python
            connector = LocalConnector("./documents")
            if connector.validate_source():
                documents = connector.load()
            ```
        """
        return self._source_path.exists() and self._source_path.is_dir()
    
    def list_documents(self) -> list[str]:
        """List all supported document files in the source directory.
        
        Returns a list of relative file paths for all supported documents
        found in the source directory. The paths are relative to the source
        directory.
        
        Returns:
            List of relative file paths for supported documents.
        
        Raises:
            ConnectorError: If the source directory cannot be accessed.
        
        Example:
            ```python
            connector = LocalConnector("./documents")
            doc_paths = connector.list_documents()
            # Returns: ["report.pdf", "notes.txt", "readme.md"]
            ```
        """
        if not self.validate_source():
            raise ConnectorError(
                f"Source directory does not exist or is not accessible: {self._source_path}"
            )
        
        document_paths: list[str] = []
        
        try:
            # Recursively find all supported files
            for file_path in self._source_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Get relative path from source directory
                    relative_path = file_path.relative_to(self._source_path)
                    document_paths.append(str(relative_path))
        except PermissionError as e:
            raise ConnectorError(
                f"Permission denied accessing directory: {self._source_path}",
                details=str(e)
            )
        except OSError as e:
            raise ConnectorError(
                f"Error accessing directory: {self._source_path}",
                details=str(e)
            )
        
        # Sort for consistent ordering
        return sorted(document_paths)
    
    def load(self) -> list[Document]:
        """Load all supported documents from the source directory.
        
        Scans the source directory for supported file types and loads each
        file into a Document object. Files with unsupported extensions are
        skipped with a warning logged.
        
        Returns:
            List of Document objects loaded from the source directory.
        
        Raises:
            ConnectorError: If the source directory cannot be accessed.
        
        Example:
            ```python
            connector = LocalConnector("./documents")
            documents = connector.load()
            for doc in documents:
                print(f"Loaded: {doc.source}")
            ```
        """
        if not self.validate_source():
            raise ConnectorError(
                f"Source directory does not exist or is not accessible: {self._source_path}"
            )
        
        documents: list[Document] = []
        
        # Get list of document paths
        doc_paths = self.list_documents()
        
        for relative_path in doc_paths:
            file_path = self._source_path / relative_path
            
            try:
                document = self._load_file(file_path)
                if document is not None:
                    documents.append(document)
            except Exception as e:
                logger.warning(
                    f"Failed to load file {relative_path}: {e}"
                )
        
        return documents
    
    def _load_file(self, file_path: Path) -> Document | None:
        """Load a single file into a Document object.
        
        Args:
            file_path: Absolute path to the file to load.
        
        Returns:
            Document object if successful, None if the file should be skipped.
        """
        extension = file_path.suffix.lower()
        
        # Check if file type is supported
        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return None
        
        # Get relative path for source identifier
        relative_path = file_path.relative_to(self._source_path)
        source = str(relative_path)
        
        # Generate document ID
        doc_id = create_document_id(source)
        
        # Detect document type
        doc_type = detect_document_type(str(file_path))
        
        # Extract content based on file type
        content = self._extract_content(file_path, extension)
        
        if content is None:
            return None
        
        # Get file stats for metadata
        file_stats = file_path.stat()
        
        # Extract metadata
        full_metadata = extract_metadata(
            str(file_path),
            content=content,
            file_stats=file_stats
        )
        
        # Create document
        return Document(
            id=doc_id,
            content=content,
            source=source,
            document_type=doc_type,
            metadata={
                "absolute_path": str(file_path),
                "file_size": file_stats.st_size,
            },
            full_metadata=full_metadata,
        )
    
    def _extract_content(self, file_path: Path, extension: str) -> str | None:
        """Extract text content from a file based on its type.
        
        Args:
            file_path: Path to the file.
            extension: File extension (lowercase, with dot).
        
        Returns:
            Extracted text content, or None if extraction failed.
        """
        try:
            if extension == ".pdf":
                return self._extract_pdf_content(file_path)
            else:
                # Text-based formats (txt, md, html)
                return self._extract_text_content(file_path)
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return None
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from a PDF file.
        
        Uses pypdf to extract text from PDF documents.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            Extracted text content from all pages.
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ConnectorError(
                "pypdf is required for PDF extraction. "
                "Install it with: pip install pypdf",
                details=str(e)
            )
        
        reader = PdfReader(str(file_path))
        text_parts: list[str] = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract text content from a text-based file.
        
        Handles encoding detection for text files using charset_normalizer
        if available, otherwise falls back to utf-8 with error handling.
        
        Args:
            file_path: Path to the text file.
        
        Returns:
            Text content of the file.
        """
        # Try to use charset_normalizer for encoding detection
        try:
            import charset_normalizer
            
            raw_data = file_path.read_bytes()
            result = charset_normalizer.from_bytes(raw_data).best()
            
            if result is not None:
                return str(result)
            else:
                # Fall back to utf-8
                return raw_data.decode("utf-8", errors="replace")
                
        except ImportError:
            # charset_normalizer not available, use simple approach
            logger.debug(
                "charset_normalizer not available, using utf-8 encoding"
            )
            
            # Try utf-8 first, then fall back to latin-1
            try:
                return file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(
                    f"UTF-8 decoding failed for {file_path}, trying latin-1"
                )
                return file_path.read_text(encoding="latin-1")