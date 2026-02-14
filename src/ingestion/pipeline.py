"""Ingestion pipeline for processing documents through cleaning and chunking.

This module provides the IngestionPipeline class that wires together connectors,
cleaners, and chunkers to process documents into chunks ready for embedding.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from src.connectors.base import BaseConnector
from src.ingestion.chunking import BaseChunker
from src.ingestion.cleaning import TextCleaner
from src.types import Chunk, Document, IngestResult

logger = logging.getLogger(__name__)


@dataclass
class IngestProgress:
    """Progress information for ingestion operations.
    
    This dataclass tracks the progress of document ingestion,
    providing real-time statistics for monitoring and reporting.
    
    Attributes:
        documents_processed: Number of documents processed so far.
        chunks_created: Total number of chunks created so far.
        current_file: The file currently being processed (optional).
        errors: List of error messages encountered.
    
    Example:
        ```python
        progress = IngestProgress(
            documents_processed=5,
            chunks_created=42,
            current_file="document.txt"
        )
        print(f"Processed {progress.documents_processed} docs")
        ```
    """
    
    documents_processed: int = 0
    chunks_created: int = 0
    current_file: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert progress to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the progress.
        """
        return {
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
            "current_file": self.current_file,
            "errors": self.errors,
        }


class IngestionPipeline:
    """Pipeline for ingesting documents through cleaning and chunking.
    
    This class wires together a document connector, text cleaner, and chunker
    to process documents from various sources into chunks ready for embedding
    and storage.
    
    Attributes:
        connector: The document connector to load documents from.
        cleaner: The text cleaner for normalizing content.
        chunker: The chunker for splitting documents into chunks.
    
    Example:
        ```python
        from src.ingestion import IngestionPipeline, TextCleaner, FixedSizeChunker
        from src.connectors import LocalConnector
        
        connector = LocalConnector("./data")
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=TextCleaner(),
            chunker=FixedSizeChunker(chunk_size=500)
        )
        result = pipeline.run()
        print(f"Ingested {result.documents_count} docs, {result.chunks_count} chunks")
        ```
    """
    
    def __init__(
        self,
        connector: BaseConnector,
        cleaner: TextCleaner,
        chunker: BaseChunker,
    ) -> None:
        """Initialize the ingestion pipeline.
        
        Args:
            connector: The document connector to load documents from.
            cleaner: The text cleaner for normalizing content.
            chunker: The chunker for splitting documents into chunks.
        """
        self.connector = connector
        self.cleaner = cleaner
        self.chunker = chunker
    
    def run(self) -> IngestResult:
        """Run the ingestion pipeline and return the result.
        
        This method loads all documents from the connector, cleans them,
        and chunks them into smaller pieces. It returns a summary of
        the ingestion operation.
        
        Returns:
            IngestResult containing counts and any errors encountered.
        
        Example:
            ```python
            result = pipeline.run()
            print(f"Processed {result.documents_count} documents")
            print(f"Created {result.chunks_count} chunks")
            if result.errors:
                print(f"Errors: {result.errors}")
            ```
        """
        documents_processed = 0
        chunks_created = 0
        errors: list[str] = []
        
        try:
            # Load documents from connector
            logger.info("Loading documents from connector...")
            documents = self.connector.load()
            logger.info(f"Loaded {len(documents)} documents")
            
            for doc in documents:
                try:
                    # Process the document
                    chunks = self._process_document(doc)
                    documents_processed += 1
                    chunks_created += len(chunks)
                    logger.debug(
                        f"Processed document {doc.id}: "
                        f"{len(chunks)} chunks created"
                    )
                except Exception as e:
                    error_msg = f"Failed to process document {doc.id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            logger.info(
                f"Ingestion complete: {documents_processed} documents, "
                f"{chunks_created} chunks"
            )
            
        except Exception as e:
            error_msg = f"Failed to load documents: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        return IngestResult(
            documents_count=documents_processed,
            chunks_count=chunks_created,
            errors=errors,
        )
    
    def run_with_progress(self) -> Iterator[IngestProgress]:
        """Run the ingestion pipeline with progress reporting.
        
        This method is useful for large datasets where you want to
        monitor progress in real-time. It yields IngestProgress objects
        after each document is processed.
        
        Yields:
            IngestProgress objects tracking the current state.
        
        Example:
            ```python
            for progress in pipeline.run_with_progress():
                print(f"Processed {progress.documents_processed} documents")
                print(f"Created {progress.chunks_created} chunks")
                if progress.current_file:
                    print(f"Current: {progress.current_file}")
            ```
        """
        progress = IngestProgress()
        
        try:
            # Load documents from connector
            logger.info("Loading documents from connector...")
            documents = self.connector.load()
            logger.info(f"Loaded {len(documents)} documents")
            
            for doc in documents:
                try:
                    # Update current file being processed
                    progress.current_file = doc.source
                    
                    # Process the document
                    chunks = self._process_document(doc)
                    progress.documents_processed += 1
                    progress.chunks_created += len(chunks)
                    
                    logger.debug(
                        f"Processed document {doc.id}: "
                        f"{len(chunks)} chunks created"
                    )
                    
                except Exception as e:
                    error_msg = f"Failed to process document {doc.id}: {str(e)}"
                    logger.error(error_msg)
                    progress.errors.append(error_msg)
                
                # Yield progress after each document
                yield progress
            
            logger.info(
                f"Ingestion complete: {progress.documents_processed} documents, "
                f"{progress.chunks_created} chunks"
            )
            
        except Exception as e:
            error_msg = f"Failed to load documents: {str(e)}"
            logger.error(error_msg)
            progress.errors.append(error_msg)
            yield progress
    
    def _process_document(self, document: Document) -> list[Chunk]:
        """Process a single document through cleaning and chunking.
        
        Args:
            document: The document to process.
        
        Returns:
            List of chunks created from the document.
        """
        # Clean the document content
        cleaned_content = self.cleaner.clean(document.content)
        
        # Prepare metadata for chunking
        metadata = {
            "doc_id": document.id,
            "source": document.source,
            "document_type": document.document_type.value if document.document_type else None,
            **document.metadata,
        }
        
        # Chunk the cleaned content
        chunks = self.chunker.chunk(cleaned_content, metadata)
        
        return chunks
    
    def process_document(self, document: Document) -> list[Chunk]:
        """Process a single document and return its chunks.
        
        This is a public method for processing individual documents
        without running the full pipeline.
        
        Args:
            document: The document to process.
        
        Returns:
            List of chunks created from the document.
        
        Example:
            ```python
            doc = Document(
                id="doc-001",
                content="Some content...",
                source="test.txt"
            )
            chunks = pipeline.process_document(doc)
            for chunk in chunks:
                print(f"Chunk: {chunk.id}")
            ```
        """
        return self._process_document(document)
    
    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return (
            f"IngestionPipeline("
            f"connector={self.connector.__class__.__name__}, "
            f"cleaner={self.cleaner.__class__.__name__}, "
            f"chunker={self.chunker.__class__.__name__})"
        )
