"""Text chunking strategies for breaking documents into processable pieces.

This module provides various chunking strategies for splitting documents
into smaller segments suitable for embedding and retrieval.
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from src.types import Chunk


@dataclass
class ChunkMetadata:
    """Metadata for tracking chunk position and relationships.
    
    Attributes:
        chunk_index: The index of this chunk in the document.
        total_chunks: Total number of chunks in the document.
        chunk_type: The type of chunking strategy used.
        has_overlap: Whether this chunk overlaps with the previous one.
        overlap_chars: Number of overlapping characters with previous chunk.
    """
    chunk_index: int
    total_chunks: int
    chunk_type: str
    has_overlap: bool = False
    overlap_chars: int = 0


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies.
    
    This class defines the interface that all chunker implementations
    must follow.
    
    Example:
        ```python
        class MyChunker(BaseChunker):
            def chunk(self, text: str, metadata: dict) -> list[Chunk]:
                # Implementation here
                pass
        ```
    """
    
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split text into chunks.
        
        Args:
            text: The text content to chunk.
            metadata: Metadata dictionary containing at least 'doc_id'.
        
        Returns:
            List of Chunk objects.
        """
        pass
    
    def _generate_chunk_id(self, document_id: str, index: int) -> str:
        """Generate a unique chunk ID.
        
        Args:
            document_id: The parent document ID.
            index: The chunk index.
        
        Returns:
            A unique chunk identifier.
        """
        return f"{document_id}-chunk-{index:04d}"


class FixedSizeChunker(BaseChunker):
    """Chunks text by character count with optional overlap.
    
    This chunker splits text into fixed-size chunks based on character
    count. It respects word boundaries by default and supports overlap
    between consecutive chunks.
    
    Attributes:
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between chunks.
        respect_word_boundaries: Whether to avoid splitting words.
    
    Example:
        ```python
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(long_text, {"doc_id": "123"})
        ```
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 0,
        respect_word_boundaries: bool = True,
    ):
        """Initialize the FixedSizeChunker.
        
        Args:
            chunk_size: Maximum characters per chunk. Must be positive.
            overlap: Characters to overlap between chunks. Must be non-negative
                and less than chunk_size.
            respect_word_boundaries: Whether to avoid splitting words.
        
        Raises:
            ValueError: If chunk_size <= 0 or overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_word_boundaries = respect_word_boundaries
    
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split text into fixed-size chunks.
        
        Args:
            text: The text content to chunk.
            metadata: Metadata dictionary containing at least 'doc_id'.
        
        Returns:
            List of Chunk objects with fixed-size content.
        """
        if not text:
            return []
        
        document_id = metadata.get("doc_id", str(uuid.uuid4()))
        chunks: list[Chunk] = []
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end and respecting word boundaries, find word boundary
            if end < len(text) and self.respect_word_boundaries:
                # Look for word boundary (space or punctuation)
                boundary = self._find_word_boundary(text, start, end)
                if boundary > start:
                    end = boundary
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            # Skip empty chunks
            if chunk_content:
                # Calculate overlap for metadata
                has_overlap = chunk_index > 0 and self.overlap > 0
                overlap_chars = self.overlap if has_overlap else 0
                
                # Create chunk metadata
                chunk_meta = ChunkMetadata(
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    chunk_type="fixed_size",
                    has_overlap=has_overlap,
                    overlap_chars=overlap_chars,
                )
                
                # Merge with provided metadata
                merged_metadata = {
                    **metadata,
                    "chunk_meta": chunk_meta.__dict__,
                }
                
                chunk = Chunk(
                    id=self._generate_chunk_id(document_id, chunk_index),
                    document_id=document_id,
                    content=chunk_content,
                    start_idx=start,
                    end_idx=end,
                    metadata=merged_metadata,
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk position (accounting for overlap)
            if self.overlap > 0 and end < len(text):
                start = end - self.overlap
            else:
                start = end
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata["chunk_meta"]["total_chunks"] = len(chunks)
        
        return chunks
    
    def _find_word_boundary(self, text: str, start: int, end: int) -> int:
        """Find a word boundary near the end position.
        
        Args:
            text: The full text.
            start: Start position of the chunk.
            end: Proposed end position.
        
        Returns:
            Position of word boundary, or original end if not found.
        """
        # Look backwards from end for a word boundary
        for i in range(end, max(start, end - 100), -1):
            if text[i] in ' \t\n\r':
                return i
        
        # No word boundary found, return original end
        return end


class SentenceChunker(BaseChunker):
    """Chunks text by sentences with size constraints.
    
    This chunker splits text into sentences and groups them to meet
    minimum and maximum size constraints. It uses natural sentence
    boundaries (periods, exclamation marks, question marks).
    
    Attributes:
        min_size: Minimum characters per chunk.
        max_size: Maximum characters per chunk.
    
    Example:
        ```python
        chunker = SentenceChunker(min_size=200, max_size=1000)
        chunks = chunker.chunk(long_text, {"doc_id": "123"})
        ```
    """
    
    # Pattern to match sentence boundaries
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # End of sentence followed by space and capital
        r'(?<=[.!?])\s+(?=[a-z])|'  # End of sentence followed by space and lowercase
        r'(?<=[.!?])(?=\s)|'        # End of sentence followed by whitespace
        r'(?<=\n)\s*(?=\S)',        # Newline followed by non-whitespace
        re.MULTILINE
    )
    
    def __init__(
        self,
        min_size: int = 200,
        max_size: int = 1000,
    ):
        """Initialize the SentenceChunker.
        
        Args:
            min_size: Minimum characters per chunk. Must be positive.
            max_size: Maximum characters per chunk. Must be greater than min_size.
        
        Raises:
            ValueError: If min_size <= 0 or max_size <= min_size.
        """
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        if max_size <= min_size:
            raise ValueError(
                f"max_size ({max_size}) must be greater than min_size ({min_size})"
            )
        
        self.min_size = min_size
        self.max_size = max_size
    
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split text into sentence-based chunks.
        
        Args:
            text: The text content to chunk.
            metadata: Metadata dictionary containing at least 'doc_id'.
        
        Returns:
            List of Chunk objects with sentence-based content.
        """
        if not text:
            return []
        
        document_id = metadata.get("doc_id", str(uuid.uuid4()))
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks: list[Chunk] = []
        current_chunk_sentences: list[tuple[str, int, int]] = []  # (sentence, start, end)
        current_size = 0
        current_start = 0
        chunk_index = 0
        
        for sentence, start_idx, end_idx in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed max_size, create a chunk
            if current_size + sentence_len > self.max_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_chunk(
                    current_chunk_sentences,
                    document_id,
                    chunk_index,
                    metadata,
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk
                current_chunk_sentences = [(sentence, start_idx, end_idx)]
                current_size = sentence_len
                current_start = start_idx
            else:
                # Add sentence to current chunk
                if not current_chunk_sentences:
                    current_start = start_idx
                current_chunk_sentences.append((sentence, start_idx, end_idx))
                current_size += sentence_len
        
        # Create final chunk if there are remaining sentences
        if current_chunk_sentences:
            chunk = self._create_chunk(
                current_chunk_sentences,
                document_id,
                chunk_index,
                metadata,
            )
            chunks.append(chunk)
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata["chunk_meta"]["total_chunks"] = len(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with their positions.
        
        Args:
            text: The text to split.
        
        Returns:
            List of tuples (sentence, start_index, end_index).
        """
        sentences: list[tuple[str, int, int]] = []
        
        # Find all sentence boundaries
        boundaries = [0]
        for match in self.SENTENCE_PATTERN.finditer(text):
            boundaries.append(match.end())
        boundaries.append(len(text))
        
        # Extract sentences
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            sentence = text[start:end].strip()
            
            if sentence:
                # Adjust start to account for stripped whitespace
                actual_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
                sentences.append((sentence, actual_start, end))
        
        # If no sentences found, return the whole text as one sentence
        if not sentences and text.strip():
            sentences.append((text.strip(), 0, len(text)))
        
        return sentences
    
    def _create_chunk(
        self,
        sentences: list[tuple[str, int, int]],
        document_id: str,
        chunk_index: int,
        metadata: dict[str, Any],
    ) -> Chunk:
        """Create a Chunk from a list of sentences.
        
        Args:
            sentences: List of (sentence, start_idx, end_idx) tuples.
            document_id: The parent document ID.
            chunk_index: The index of this chunk.
            metadata: Additional metadata to include.
        
        Returns:
            A Chunk object.
        """
        content = " ".join(s[0] for s in sentences)
        start_idx = sentences[0][1]
        end_idx = sentences[-1][2]
        
        chunk_meta = ChunkMetadata(
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            chunk_type="sentence",
            has_overlap=False,
            overlap_chars=0,
        )
        
        merged_metadata = {
            **metadata,
            "chunk_meta": chunk_meta.__dict__,
            "sentence_count": len(sentences),
        }
        
        return Chunk(
            id=self._generate_chunk_id(document_id, chunk_index),
            document_id=document_id,
            content=content,
            start_idx=start_idx,
            end_idx=end_idx,
            metadata=merged_metadata,
        )
