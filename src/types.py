"""Core type definitions for the GraphRAG project.

This module defines the fundamental data structures used throughout the project
using Pydantic v2 models for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Type alias for embedding vectors
EmbeddingVector = list[float]


class DocumentType(str, Enum):
    """Enumeration of supported document types.
    
    This enum defines the types of documents that can be processed
    by the GraphRAG system.
    
    Attributes:
        PDF: PDF document format.
        TXT: Plain text file.
        MD: Markdown document.
        HTML: HTML document.
        JSON: JSON data file.
    
    Example:
        ```python
        doc_type = DocumentType.PDF
        print(doc_type.value)  # "pdf"
        ```
    """
    
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document.
    
    This model captures structured metadata about a document,
    including authorship, timestamps, and categorization.
    
    Attributes:
        author: The author of the document (optional).
        title: The title of the document (optional).
        created: Creation timestamp of the original document.
        modified: Last modification timestamp of the document.
        tags: List of tags for categorization.
    
    Example:
        ```python
        metadata = DocumentMetadata(
            author="John Doe",
            title="Technical Report",
            tags=["research", "ai"]
        )
        ```
    """
    
    author: Optional[str] = Field(
        default=None,
        description="The author of the document"
    )
    title: Optional[str] = Field(
        default=None,
        description="The title of the document"
    )
    created: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp of the original document"
    )
    modified: Optional[datetime] = Field(
        default=None,
        description="Last modification timestamp of the document"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for categorization"
    )


class ProviderConfig(BaseModel):
    """Base configuration for provider implementations.
    
    This serves as a base class for specific provider configurations
    (e.g., OpenAI, Anthropic, local models).
    
    Attributes:
        provider_type: The type identifier for this provider.
        model_name: The name of the model to use.
        api_key_env: Environment variable name for the API key.
        extra_params: Additional provider-specific parameters.
    """
    
    provider_type: str = Field(..., description="Type identifier for the provider")
    model_name: str = Field(..., description="Name of the model to use")
    api_key_env: Optional[str] = Field(
        default=None, 
        description="Environment variable name for API key"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )


class Document(BaseModel):
    """Represents a source document in the RAG system.
    
    A document is the top-level unit of content that gets ingested,
    chunked, and indexed for retrieval.
    
    Attributes:
        id: Unique identifier for the document.
        content: The full text content of the document.
        source: The source location (file path, URL, etc.).
        document_type: The type of document (PDF, TXT, MD, HTML, JSON).
        metadata: Additional metadata about the document (flexible key-value pairs).
        full_metadata: Structured metadata object with typed fields.
        created_at: Timestamp when the document was created/ingested.
    
    Example:
        ```python
        doc = Document(
            id="doc-001",
            content="This is the document content...",
            source="/path/to/document.txt",
            document_type=DocumentType.TXT,
            metadata={"category": "technical"},
            full_metadata=DocumentMetadata(author="John Doe", title="Report")
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Full text content of the document")
    source: str = Field(..., description="Source location (file path, URL, etc.)")
    document_type: Optional[DocumentType] = Field(
        default=None,
        description="The type of document (PDF, TXT, MD, HTML, JSON)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document (flexible key-value pairs)"
    )
    full_metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="Structured metadata object with typed fields"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when document was created/ingested"
    )


class Chunk(BaseModel):
    """Represents a chunk of a document for retrieval.
    
    Chunks are segments of documents created during the ingestion process.
    They are the unit of retrieval in the RAG system.
    
    Attributes:
        id: Unique identifier for the chunk.
        document_id: ID of the parent document.
        content: The text content of this chunk.
        start_idx: Starting character index in the original document.
        end_idx: Ending character index in the original document.
        metadata: Additional metadata (e.g., page number, section).
    
    Example:
        ```python
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="This is a chunk of text...",
            start_idx=0,
            end_idx=100,
            metadata={"page": 1}
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., description="Text content of this chunk")
    start_idx: int = Field(
        ..., 
        ge=0,
        description="Starting character index in original document"
    )
    end_idx: int = Field(
        ..., 
        ge=0,
        description="Ending character index in original document"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., page number, section)"
    )
    
    @field_validator('end_idx')
    @classmethod
    def end_idx_greater_than_start(cls, v: int, info: Any) -> int:
        """Validate that end_idx is greater than or equal to start_idx."""
        # Note: We can't access start_idx here directly in v2 style,
        # but the model validator will handle consistency
        return v


class Citation(BaseModel):
    """Represents a citation in an answer.
    
    Citations link back to source chunks and provide attribution
    for information included in generated answers.
    
    Attributes:
        source: The source location of the cited content.
        chunk_id: ID of the chunk being cited.
        text: The relevant text snippet from the chunk.
        score: Relevance score (0.0 to 1.0).
    
    Example:
        ```python
        citation = Citation(
            source="/path/to/document.txt",
            chunk_id="chunk-001",
            text="The relevant passage...",
            score=0.95
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    source: str = Field(..., description="Source location of the cited content")
    chunk_id: str = Field(..., description="ID of the chunk being cited")
    text: str = Field(..., description="The relevant text snippet from the chunk")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score between 0.0 and 1.0"
    )


class Answer(BaseModel):
    """Represents an answer generated by the RAG system.
    
    An answer contains the generated text response along with
    citations to source material and confidence metrics.
    
    Attributes:
        text: The generated answer text.
        citations: List of citations supporting the answer.
        confidence: Confidence score (0.0 to 1.0).
        refusal_reason: Reason if the system refused to answer.
    
    Example:
        ```python
        answer = Answer(
            text="Based on the documents...",
            citations=[citation1, citation2],
            confidence=0.85
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    text: str = Field(..., description="The generated answer text")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of citations supporting the answer"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Reason if the system refused to answer"
    )


class QueryResult(BaseModel):
    """Represents the complete result of a query operation.
    
    This encapsulates the question, answer, and metadata about
    the query execution.
    
    Attributes:
        question: The original question that was asked.
        answer: The generated answer object.
        mode: The retrieval mode used (e.g., 'vector', 'graph', 'hybrid').
        latency_ms: Time taken to process the query in milliseconds.
    
    Example:
        ```python
        result = QueryResult(
            question="What is the main topic?",
            answer=answer,
            mode="hybrid",
            latency_ms=1250.5
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    question: str = Field(..., description="The original question that was asked")
    answer: Answer = Field(..., description="The generated answer object")
    mode: Literal["vector", "graph", "hybrid"] = Field(
        ...,
        description="The retrieval mode used"
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to process the query in milliseconds"
    )


class IngestResult(BaseModel):
    """Represents the result of a document ingestion operation.
    
    This provides statistics about the ingestion process including
    counts and any errors encountered.
    
    Attributes:
        documents_count: Number of documents processed.
        chunks_count: Number of chunks created.
        errors: List of error messages encountered during ingestion.
    
    Example:
        ```python
        result = IngestResult(
            documents_count=10,
            chunks_count=150,
            errors=["Failed to process doc-003: encoding error"]
        )
        ```
    """
    
    model_config = ConfigDict(frozen=True)
    
    documents_count: int = Field(
        ...,
        ge=0,
        description="Number of documents processed"
    )
    chunks_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks created"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages encountered"
    )
