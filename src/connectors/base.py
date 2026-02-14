"""Base connector interface for document sources.

This module defines the abstract base class that all document connector
implementations must follow, ensuring a consistent interface for loading
documents from various sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.types import Document


class BaseConnector(ABC):
    """Abstract base class for document connectors.
    
    Document connectors are responsible for loading documents from various
    sources (local files, cloud storage, databases, APIs, etc.) and converting
    them to the standard Document format used by the GraphRAG system.
    
    All connector implementations must inherit from this class and implement
    the abstract methods defined here.
    
    Attributes:
        source: The source location (file path, URL, connection string, etc.).
    
    Example:
        ```python
        class FileConnector(BaseConnector):
            def __init__(self, source: str):
                self.source = source
            
            def load(self) -> list[Document]:
                # Load documents from files
                pass
            
            def list_documents(self) -> list[str]:
                # List available files
                pass
        ```
    """
    
    def __init__(self, source: str) -> None:
        """Initialize the connector with a source location.
        
        Args:
            source: The source location (file path, URL, connection string, etc.).
        """
        self._source = source
    
    @property
    def source(self) -> str:
        """Return the source location for this connector.
        
        Returns:
            The source location string.
        """
        return self._source
    
    @abstractmethod
    def load(self) -> list[Document]:
        """Load all documents from the source.
        
        This method must be implemented by all subclasses to load documents
        from the configured source and return them as a list of Document objects.
        
        Returns:
            A list of Document objects loaded from the source.
        
        Raises:
            ConnectorError: If the source cannot be accessed or documents
                cannot be loaded.
        
        Example:
            ```python
            connector = MyConnector("/path/to/documents")
            documents = connector.load()
            for doc in documents:
                print(f"Loaded: {doc.id}")
            ```
        """
        pass
    
    @abstractmethod
    def list_documents(self) -> list[str]:
        """List available document IDs or names from the source.
        
        This method must be implemented by all subclasses to return a list
        of identifiers for documents available at the source. This can be
        used to preview what documents are available before loading them.
        
        Returns:
            A list of document IDs or names available at the source.
        
        Raises:
            ConnectorError: If the source cannot be accessed.
        
        Example:
            ```python
            connector = MyConnector("/path/to/documents")
            doc_ids = connector.list_documents()
            print(f"Available documents: {doc_ids}")
            ```
        """
        pass
    
    def validate_source(self) -> bool:
        """Check if the source is accessible and valid.
        
        This method provides a default implementation that checks if a
        file path exists. Subclasses should override this method to provide
        appropriate validation for their specific source type (e.g., checking
        network connectivity for remote sources, validating credentials, etc.).
        
        Returns:
            True if the source is accessible and valid, False otherwise.
        
        Example:
            ```python
            connector = MyConnector("/path/to/documents")
            if connector.validate_source():
                documents = connector.load()
            else:
                print("Source is not accessible")
            ```
        """
        path = Path(self._source)
        return path.exists()
