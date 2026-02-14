"""Custom exceptions for the GraphRAG project.

This module defines a hierarchy of exceptions used throughout the project
to provide clear and specific error handling.
"""


class RAGError(Exception):
    """Base exception for all RAG-related errors.
    
    This serves as the parent class for all custom exceptions in the
    GraphRAG project, allowing for broad exception catching when needed.
    
    Attributes:
        message: Human-readable error description.
        details: Optional additional details about the error.
    
    Example:
        ```python
        try:
            # Some RAG operation
            pass
        except RAGError as e:
            print(f"RAG operation failed: {e.message}")
        ```
    """
    
    def __init__(self, message: str, details: str | None = None) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error description.
            details: Optional additional details about the error.
        """
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConnectorError(RAGError):
    """Exception raised when a document connector fails.
    
    This exception is raised when there are issues with document
    connectors, such as file access errors, unsupported formats,
    or network issues when fetching remote documents.
    
    Example:
        ```python
        raise ConnectorError(
            "Failed to read document",
            details="File not found: /path/to/document.pdf"
        )
        ```
    """
    pass


class IngestionError(RAGError):
    """Exception raised during document ingestion failures.
    
    This exception is raised when the ingestion pipeline fails,
    such as chunking errors, parsing failures, or validation issues.
    
    Example:
        ```python
        raise IngestionError(
            "Failed to ingest document",
            details="Document exceeds maximum size limit"
        )
        ```
    """
    pass


class EmbeddingError(RAGError):
    """Exception raised when embedding generation fails.
    
    This exception is raised when there are issues with the embedding
    model, such as API errors, rate limits, or invalid inputs.
    
    Example:
        ```python
        raise EmbeddingError(
            "Failed to generate embeddings",
            details="OpenAI API rate limit exceeded"
        )
        ```
    """
    pass


class VectorStoreError(RAGError):
    """Exception raised when vector store operations fail.
    
    This exception is raised when there are issues with the vector
    database, such as connection errors, query failures, or index issues.
    
    Example:
        ```python
        raise VectorStoreError(
            "Failed to query vector store",
            details="Connection refused to Pinecone"
        )
        ```
    """
    pass


class RetrievalError(RAGError):
    """Exception raised when retrieval operations fail.
    
    This exception is raised when there are issues during the retrieval
    phase, such as query parsing errors or ranking failures.
    
    Example:
        ```python
        raise RetrievalError(
            "Failed to retrieve documents",
            details="Invalid query syntax"
        )
        ```
    """
    pass


class LLMError(RAGError):
    """Exception raised when LLM operations fail.
    
    This exception is raised when there are issues with the language
    model, such as API errors, context length exceeded, or generation
    failures.
    
    Example:
        ```python
        raise LLMError(
            "Failed to generate response",
            details="Context length exceeded (8192 tokens)"
        )
        ```
    """
    pass


class ConfigurationError(RAGError):
    """Exception raised when configuration is invalid.
    
    This exception is raised when there are issues with the configuration,
    such as missing required settings, invalid values, or file not found.
    
    Example:
        ```python
        raise ConfigurationError(
            "Invalid configuration",
            details="Missing required field: openai_api_key"
        )
        ```
    """
    pass
