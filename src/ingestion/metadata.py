"""Metadata extraction for document chunks.

This module provides functionality for extracting and managing metadata
associated with document chunks, including position information, context,
and unique identifier generation.
"""

from typing import Any

from src.types import Chunk, Document


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from document ID and chunk index.
    
    The ID format is: "{document_id}_chunk_{index}" which provides a
    consistent, predictable identifier for chunks.
    
    Args:
        document_id: The unique identifier of the parent document.
        chunk_index: The zero-based index of the chunk within the document.
    
    Returns:
        A deterministic chunk identifier string.
    
    Example:
        ```python
        chunk_id = generate_chunk_id("doc_123", 0)  # "doc_123_chunk_0"
        chunk_id = generate_chunk_id("doc_123", 5)  # "doc_123_chunk_5"
        ```
    """
    return f"{document_id}_chunk_{chunk_index}"


def extract_position_info(chunk: Chunk) -> dict[str, Any]:
    """Extract position information from a chunk.
    
    This function extracts positional metadata from a chunk including
    page number, paragraph information, and character position within
    the document.
    
    Args:
        chunk: The Chunk object to extract position info from.
    
    Returns:
        A dictionary containing position information with keys:
            - start_char: Starting character index in the document
            - end_char: Ending character index in the document
            - char_count: Number of characters in the chunk
            - page: Page number if available (from chunk metadata)
            - paragraph: Paragraph number if available (from chunk metadata)
            - position: Relative position in document (0.0 to 1.0)
    
    Example:
        ```python
        position_info = extract_position_info(chunk)
        # Returns: {
        #     "start_char": 0,
        #     "end_char": 500,
        #     "char_count": 500,
        #     "page": 1,
        #     "paragraph": 0,
        #     "position": 0.0
        # }
        ```
    """
    # Get chunk metadata
    chunk_meta = chunk.metadata.get("chunk_meta", {})
    
    # Calculate position relative to document (requires document length)
    # This will be 0.0 if we don't have document context
    position = 0.0
    if "relative_position" in chunk.metadata:
        position = chunk.metadata["relative_position"]
    
    return {
        "start_char": chunk.start_idx,
        "end_char": chunk.end_idx,
        "char_count": len(chunk.content),
        "page": chunk.metadata.get("page"),
        "paragraph": chunk.metadata.get("paragraph"),
        "position": position,
        "chunk_index": chunk_meta.get("chunk_index"),
        "total_chunks": chunk_meta.get("total_chunks"),
        "chunk_type": chunk_meta.get("chunk_type"),
        "has_overlap": chunk_meta.get("has_overlap", False),
        "overlap_chars": chunk_meta.get("overlap_chars", 0),
    }


def extract_context(chunk: Chunk, document: Document) -> dict[str, Any]:
    """Extract surrounding context for a chunk from its document.
    
    This function extracts text that appears before and after the chunk
    in the original document, providing context for retrieval and display.
    
    Args:
        chunk: The Chunk object to extract context for.
        document: The parent Document object.
    
    Returns:
        A dictionary containing context information with keys:
            - before: Text appearing immediately before the chunk (up to 200 chars)
            - after: Text appearing immediately after the chunk (up to 200 chars)
            - document_source: The source of the document
            - document_type: The type of the document
    
    Example:
        ```python
        context = extract_context(chunk, document)
        # Returns: {
        #     "before": "...previous text...",
        #     "after": "...following text...",
        #     "document_source": "/path/to/document.txt",
        #     "document_type": "txt"
        # }
        ```
    """
    # Default context window size
    context_window = 200
    
    # Extract text before the chunk
    before_start = max(0, chunk.start_idx - context_window)
    before_text = document.content[before_start:chunk.start_idx]
    
    # Add ellipsis if we're not at the start
    if before_start > 0:
        before_text = "..." + before_text
    
    # Extract text after the chunk
    after_end = min(len(document.content), chunk.end_idx + context_window)
    after_text = document.content[chunk.end_idx:after_end]
    
    # Add ellipsis if we're not at the end
    if after_end < len(document.content):
        after_text = after_text + "..."
    
    return {
        "before": before_text,
        "after": after_text,
        "document_source": document.source,
        "document_type": document.document_type.value if document.document_type else None,
        "document_id": document.id,
    }


class ChunkMetadataExtractor:
    """Extracts comprehensive metadata from document chunks.
    
    This class provides a unified interface for extracting all metadata
    associated with a chunk, combining position information, context,
    and document-level metadata.
    
    Attributes:
        context_window: Number of characters to include as context before/after chunk.
    
    Example:
        ```python
        extractor = ChunkMetadataExtractor(context_window=300)
        metadata = extractor.extract(chunk, document)
        # metadata contains position_info, context, and document metadata
        ```
    """
    
    def __init__(self, context_window: int = 200):
        """Initialize the ChunkMetadataExtractor.
        
        Args:
            context_window: Number of characters to extract as context before/after
                the chunk. Defaults to 200.
        
        Raises:
            ValueError: If context_window is negative.
        """
        if context_window < 0:
            raise ValueError(f"context_window must be non-negative, got {context_window}")
        self.context_window = context_window
    
    def extract(self, chunk: Chunk, document: Document) -> dict[str, Any]:
        """Extract comprehensive metadata from a chunk.
        
        This method combines all metadata extraction into a single dictionary,
        including position information, surrounding context, and document-level
        metadata.
        
        Args:
            chunk: The Chunk object to extract metadata from.
            document: The parent Document object.
        
        Returns:
            A dictionary containing all extracted metadata with keys:
                - chunk_id: The unique identifier of the chunk
                - document_id: The parent document ID
                - position: Position information dict
                - context: Surrounding context dict
                - document_metadata: Document-level metadata
                - content_length: Length of chunk content
                - content_hash: Hash of chunk content for deduplication
        
        Example:
            ```python
            extractor = ChunkMetadataExtractor()
            metadata = extractor.extract(chunk, document)
            print(metadata["chunk_id"])  # "doc_123_chunk_0"
            ```
        """
        # Extract position information
        position_info = extract_position_info(chunk)
        
        # Calculate relative position in document
        if len(document.content) > 0:
            position_info["position"] = chunk.start_idx / len(document.content)
        
        # Extract context with configured window size
        context = self._extract_context_with_window(chunk, document)
        
        # Get document metadata
        doc_metadata = self._extract_document_metadata(document)
        
        # Generate content hash for deduplication
        content_hash = hash(chunk.content) if chunk.content else 0
        
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "position": position_info,
            "context": context,
            "document_metadata": doc_metadata,
            "content_length": len(chunk.content),
            "content_hash": content_hash,
        }
    
    def _extract_context_with_window(self, chunk: Chunk, document: Document) -> dict[str, Any]:
        """Extract context with the configured window size.
        
        Args:
            chunk: The Chunk object to extract context for.
            document: The parent Document object.
        
        Returns:
            Dictionary with before/after context text.
        """
        # Extract text before the chunk
        before_start = max(0, chunk.start_idx - self.context_window)
        before_text = document.content[before_start:chunk.start_idx]
        
        # Add ellipsis if we're not at the start
        if before_start > 0:
            before_text = "..." + before_text
        
        # Extract text after the chunk
        after_end = min(len(document.content), chunk.end_idx + self.context_window)
        after_text = document.content[chunk.end_idx:after_end]
        
        # Add ellipsis if we're not at the end
        if after_end < len(document.content):
            after_text = after_text + "..."
        
        return {
            "before": before_text,
            "after": after_text,
            "document_source": document.source,
            "document_type": document.document_type.value if document.document_type else None,
        }
    
    def _extract_document_metadata(self, document: Document) -> dict[str, Any]:
        """Extract relevant metadata from the document.
        
        Args:
            document: The Document object to extract metadata from.
        
        Returns:
            Dictionary with document-level metadata.
        """
        result = {
            "source": document.source,
            "document_type": document.document_type.value if document.document_type else None,
            "created_at": document.created_at.isoformat() if document.created_at else None,
        }
        
        # Add full metadata if available
        if document.full_metadata:
            result["author"] = document.full_metadata.author
            result["title"] = document.full_metadata.title
            result["tags"] = document.full_metadata.tags
            if document.full_metadata.created:
                result["document_created"] = document.full_metadata.created.isoformat()
            if document.full_metadata.modified:
                result["document_modified"] = document.full_metadata.modified.isoformat()
        
        # Add any additional metadata from the document
        if document.metadata:
            result["custom"] = document.metadata
        
        return result
