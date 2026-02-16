"""Pydantic schemas for API request and response models.

This module defines the request and response schemas used by the
FastAPI endpoints for validation and serialization.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.types import Citation


class QueryRequest(BaseModel):
    """Request schema for the query endpoint.
    
    Attributes:
        question: The user's question to be answered.
        mode: The retrieval mode to use (qna, vector, hybrid).
        k: Number of documents to retrieve.
    
    Example:
        ```python
        request = QueryRequest(
            question="What is federalism?",
            mode="qna",
            k=5
        )
        ```
    """
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question to be answered"
    )
    mode: Literal["qna", "vector", "hybrid"] = Field(
        default="qna",
        description="The retrieval mode to use"
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    
    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, v: str) -> str:
        """Validate that question is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("question must be a non-empty string")
        return v


class QueryResponse(BaseModel):
    """Response schema for the query endpoint.
    
    Attributes:
        answer: The generated answer text, or None if refused.
        citations: List of citations supporting the answer.
        confidence: Confidence score between 0.0 and 1.0.
        refusal_reason: Reason for refusal if the system refused to answer.
        latency_ms: Time taken to process the query in milliseconds.
    
    Example:
        ```python
        response = QueryResponse(
            answer="Federalism is a system of government...",
            citations=[citation1, citation2],
            confidence=0.92,
            refusal_reason=None,
            latency_ms=1234
        )
        ```
    """
    
    answer: Optional[str] = Field(
        default=None,
        description="The generated answer text, or None if refused"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of citations supporting the answer"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Reason for refusal if the system refused to answer"
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to process the query in milliseconds"
    )


class IngestRequest(BaseModel):
    """Request schema for the ingest endpoint.
    
    Attributes:
        corpus: The name of the corpus to ingest.
        async_ingest: Whether to run ingestion asynchronously (optional).
    
    Example:
        ```python
        request = IngestRequest(
            corpus="public_domain_gov",
            async_ingest=False
        )
        ```
    """
    
    corpus: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The name of the corpus to ingest"
    )
    async_ingest: bool = Field(
        default=False,
        description="Whether to run ingestion asynchronously"
    )
    
    @field_validator("corpus")
    @classmethod
    def corpus_must_not_be_whitespace(cls, v: str) -> str:
        """Validate that corpus is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("corpus must be a non-empty string")
        return v


class IngestResponse(BaseModel):
    """Response schema for the ingest endpoint.
    
    Attributes:
        status: The status of the ingestion (completed, failed, in_progress).
        documents_count: Number of documents processed.
        chunks_count: Number of chunks created.
        errors: List of error messages encountered.
    
    Example:
        ```python
        response = IngestResponse(
            status="completed",
            documents_count=42,
            chunks_count=1250,
            errors=[]
        )
        ```
    """
    
    status: Literal["completed", "failed", "in_progress"] = Field(
        default="completed",
        description="The status of the ingestion"
    )
    documents_count: int = Field(
        default=0,
        ge=0,
        description="Number of documents processed"
    )
    chunks_count: int = Field(
        default=0,
        ge=0,
        description="Number of chunks created"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages encountered"
    )
