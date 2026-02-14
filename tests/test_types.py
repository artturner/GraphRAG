"""Tests for core type definitions and exceptions.

This module contains comprehensive tests for the Pydantic models
defined in src/types.py and exceptions in src/exceptions.py.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.exceptions import (
    ConfigurationError,
    ConnectorError,
    EmbeddingError,
    IngestionError,
    LLMError,
    RAGError,
    RetrievalError,
    VectorStoreError,
)
from src.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    EmbeddingVector,
    IngestResult,
    ProviderConfig,
    QueryResult,
)


class TestDocument:
    """Tests for Document model."""
    
    def test_document_creation_basic(self) -> None:
        """Test basic document creation with required fields."""
        doc = Document(
            id="doc-001",
            content="This is test content.",
            source="/path/to/document.txt",
        )
        
        assert doc.id == "doc-001"
        assert doc.content == "This is test content."
        assert doc.source == "/path/to/document.txt"
        assert doc.metadata == {}
        assert isinstance(doc.created_at, datetime)
    
    def test_document_creation_with_metadata(self) -> None:
        """Test document creation with metadata."""
        doc = Document(
            id="doc-002",
            content="Content with metadata.",
            source="https://example.com/doc",
            metadata={"author": "John Doe", "category": "technical"},
        )
        
        assert doc.metadata["author"] == "John Doe"
        assert doc.metadata["category"] == "technical"
    
    def test_document_creation_with_custom_timestamp(self) -> None:
        """Test document creation with custom timestamp."""
        custom_time = datetime(2024, 1, 15, 10, 30, 0)
        doc = Document(
            id="doc-003",
            content="Content with timestamp.",
            source="/path/to/doc.txt",
            created_at=custom_time,
        )
        
        assert doc.created_at == custom_time
    
    def test_document_is_frozen(self) -> None:
        """Test that Document model is immutable (frozen)."""
        doc = Document(
            id="doc-004",
            content="Immutable content.",
            source="/path/to/doc.txt",
        )
        
        with pytest.raises(ValidationError):
            doc.id = "new-id"  # type: ignore[misc]
    
    def test_document_missing_required_fields(self) -> None:
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Document(id="doc-005")  # type: ignore[call-arg]
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "content" in error_fields
        assert "source" in error_fields
    
    def test_document_serialization(self) -> None:
        """Test document serialization to dict."""
        doc = Document(
            id="doc-006",
            content="Serialization test.",
            source="/path/to/doc.txt",
            metadata={"key": "value"},
        )
        
        doc_dict = doc.model_dump()
        assert doc_dict["id"] == "doc-006"
        assert doc_dict["content"] == "Serialization test."
        assert doc_dict["metadata"] == {"key": "value"}


class TestChunk:
    """Tests for Chunk model."""
    
    def test_chunk_creation_basic(self) -> None:
        """Test basic chunk creation."""
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="This is a chunk of text.",
            start_idx=0,
            end_idx=23,
        )
        
        assert chunk.id == "chunk-001"
        assert chunk.document_id == "doc-001"
        assert chunk.content == "This is a chunk of text."
        assert chunk.start_idx == 0
        assert chunk.end_idx == 23
    
    def test_chunk_creation_with_metadata(self) -> None:
        """Test chunk creation with metadata."""
        chunk = Chunk(
            id="chunk-002",
            document_id="doc-001",
            content="Chunk with metadata.",
            start_idx=100,
            end_idx=120,
            metadata={"page": 1, "section": "Introduction"},
        )
        
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["section"] == "Introduction"
    
    def test_chunk_negative_start_idx_raises_error(self) -> None:
        """Test that negative start_idx raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Chunk(
                id="chunk-003",
                document_id="doc-001",
                content="Invalid chunk.",
                start_idx=-1,
                end_idx=10,
            )
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "start_idx" in error_fields
    
    def test_chunk_negative_end_idx_raises_error(self) -> None:
        """Test that negative end_idx raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Chunk(
                id="chunk-004",
                document_id="doc-001",
                content="Invalid chunk.",
                start_idx=0,
                end_idx=-5,
            )
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "end_idx" in error_fields
    
    def test_chunk_is_frozen(self) -> None:
        """Test that Chunk model is immutable (frozen)."""
        chunk = Chunk(
            id="chunk-005",
            document_id="doc-001",
            content="Immutable chunk.",
            start_idx=0,
            end_idx=16,
        )
        
        with pytest.raises(ValidationError):
            chunk.content = "modified"  # type: ignore[misc]


class TestCitation:
    """Tests for Citation model."""
    
    def test_citation_creation(self) -> None:
        """Test basic citation creation."""
        citation = Citation(
            source="/path/to/document.txt",
            chunk_id="chunk-001",
            text="The cited text passage.",
            score=0.95,
        )
        
        assert citation.source == "/path/to/document.txt"
        assert citation.chunk_id == "chunk-001"
        assert citation.text == "The cited text passage."
        assert citation.score == 0.95
    
    def test_citation_score_validation_valid(self) -> None:
        """Test that valid scores are accepted."""
        # Test boundary values
        citation_min = Citation(
            source="source",
            chunk_id="chunk",
            text="text",
            score=0.0,
        )
        assert citation_min.score == 0.0
        
        citation_max = Citation(
            source="source",
            chunk_id="chunk",
            text="text",
            score=1.0,
        )
        assert citation_max.score == 1.0
    
    def test_citation_score_validation_invalid_high(self) -> None:
        """Test that scores above 1.0 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Citation(
                source="source",
                chunk_id="chunk",
                text="text",
                score=1.5,
            )
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "score" in error_fields
    
    def test_citation_score_validation_invalid_low(self) -> None:
        """Test that scores below 0.0 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Citation(
                source="source",
                chunk_id="chunk",
                text="text",
                score=-0.1,
            )
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "score" in error_fields
    
    def test_citation_is_frozen(self) -> None:
        """Test that Citation model is immutable (frozen)."""
        citation = Citation(
            source="source",
            chunk_id="chunk",
            text="text",
            score=0.5,
        )
        
        with pytest.raises(ValidationError):
            citation.score = 0.9  # type: ignore[misc]


class TestAnswer:
    """Tests for Answer model."""
    
    def test_answer_creation_basic(self) -> None:
        """Test basic answer creation."""
        answer = Answer(
            text="This is the answer.",
            confidence=0.85,
        )
        
        assert answer.text == "This is the answer."
        assert answer.citations == []
        assert answer.confidence == 0.85
        assert answer.refusal_reason is None
    
    def test_answer_with_citations(self) -> None:
        """Test answer creation with citations."""
        citation1 = Citation(
            source="source1",
            chunk_id="chunk1",
            text="citation text 1",
            score=0.9,
        )
        citation2 = Citation(
            source="source2",
            chunk_id="chunk2",
            text="citation text 2",
            score=0.8,
        )
        
        answer = Answer(
            text="Answer with citations.",
            citations=[citation1, citation2],
            confidence=0.9,
        )
        
        assert len(answer.citations) == 2
        assert answer.citations[0].chunk_id == "chunk1"
        assert answer.citations[1].chunk_id == "chunk2"
    
    def test_answer_with_refusal_reason(self) -> None:
        """Test answer creation with refusal reason."""
        answer = Answer(
            text="I cannot answer this question.",
            confidence=0.0,
            refusal_reason="Insufficient context in provided documents.",
        )
        
        assert answer.refusal_reason == "Insufficient context in provided documents."
    
    def test_answer_confidence_validation(self) -> None:
        """Test that confidence must be between 0 and 1."""
        # Valid confidence
        answer = Answer(text="text", confidence=0.5)
        assert answer.confidence == 0.5
        
        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            Answer(text="text", confidence=1.5)
        
        # Invalid confidence (negative)
        with pytest.raises(ValidationError):
            Answer(text="text", confidence=-0.1)
    
    def test_answer_is_frozen(self) -> None:
        """Test that Answer model is immutable (frozen)."""
        answer = Answer(text="text", confidence=0.5)
        
        with pytest.raises(ValidationError):
            answer.text = "modified"  # type: ignore[misc]


class TestQueryResult:
    """Tests for QueryResult model."""
    
    def test_query_result_creation(self) -> None:
        """Test basic query result creation."""
        answer = Answer(text="Answer text.", confidence=0.9)
        result = QueryResult(
            question="What is the topic?",
            answer=answer,
            mode="vector",
            latency_ms=1250.5,
        )
        
        assert result.question == "What is the topic?"
        assert result.answer.text == "Answer text."
        assert result.mode == "vector"
        assert result.latency_ms == 1250.5
    
    def test_query_result_modes(self) -> None:
        """Test that all valid modes are accepted."""
        answer = Answer(text="text", confidence=0.5)
        
        for mode in ["vector", "graph", "hybrid"]:
            result = QueryResult(
                question="question",
                answer=answer,
                mode=mode,  # type: ignore[arg-type]
                latency_ms=100.0,
            )
            assert result.mode == mode
    
    def test_query_result_invalid_mode(self) -> None:
        """Test that invalid mode raises ValidationError."""
        answer = Answer(text="text", confidence=0.5)
        
        with pytest.raises(ValidationError) as exc_info:
            QueryResult(
                question="question",
                answer=answer,
                mode="invalid",  # type: ignore[arg-type]
                latency_ms=100.0,
            )
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "mode" in error_fields
    
    def test_query_result_negative_latency(self) -> None:
        """Test that negative latency raises ValidationError."""
        answer = Answer(text="text", confidence=0.5)
        
        with pytest.raises(ValidationError) as exc_info:
            QueryResult(
                question="question",
                answer=answer,
                mode="vector",
                latency_ms=-100.0,
            )
        
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "latency_ms" in error_fields
    
    def test_query_result_serialization(self) -> None:
        """Test QueryResult serialization."""
        answer = Answer(text="Answer.", confidence=0.9)
        result = QueryResult(
            question="Question?",
            answer=answer,
            mode="hybrid",
            latency_ms=500.0,
        )
        
        result_dict = result.model_dump()
        assert result_dict["question"] == "Question?"
        assert result_dict["mode"] == "hybrid"
        assert result_dict["latency_ms"] == 500.0
        assert "answer" in result_dict


class TestIngestResult:
    """Tests for IngestResult model."""
    
    def test_ingest_result_creation_basic(self) -> None:
        """Test basic ingest result creation."""
        result = IngestResult(
            documents_count=10,
            chunks_count=150,
        )
        
        assert result.documents_count == 10
        assert result.chunks_count == 150
        assert result.errors == []
    
    def test_ingest_result_with_errors(self) -> None:
        """Test ingest result with errors."""
        result = IngestResult(
            documents_count=8,
            chunks_count=120,
            errors=["Failed to process doc-003", "Encoding error in doc-007"],
        )
        
        assert len(result.errors) == 2
        assert "Failed to process doc-003" in result.errors
    
    def test_ingest_result_negative_counts_invalid(self) -> None:
        """Test that negative counts raise ValidationError."""
        with pytest.raises(ValidationError):
            IngestResult(documents_count=-1, chunks_count=10)
        
        with pytest.raises(ValidationError):
            IngestResult(documents_count=10, chunks_count=-1)
    
    def test_ingest_result_is_frozen(self) -> None:
        """Test that IngestResult model is immutable (frozen)."""
        result = IngestResult(documents_count=5, chunks_count=50)
        
        with pytest.raises(ValidationError):
            result.documents_count = 10  # type: ignore[misc]


class TestProviderConfig:
    """Tests for ProviderConfig model."""
    
    def test_provider_config_creation(self) -> None:
        """Test basic provider config creation."""
        config = ProviderConfig(
            provider_type="openai",
            model_name="gpt-4",
        )
        
        assert config.provider_type == "openai"
        assert config.model_name == "gpt-4"
        assert config.api_key_env is None
        assert config.extra_params == {}
    
    def test_provider_config_with_all_fields(self) -> None:
        """Test provider config with all fields."""
        config = ProviderConfig(
            provider_type="anthropic",
            model_name="claude-3-opus",
            api_key_env="ANTHROPIC_API_KEY",
            extra_params={"temperature": 0.7, "max_tokens": 4096},
        )
        
        assert config.api_key_env == "ANTHROPIC_API_KEY"
        assert config.extra_params["temperature"] == 0.7
        assert config.extra_params["max_tokens"] == 4096


class TestEmbeddingVector:
    """Tests for EmbeddingVector type alias."""
    
    def test_embedding_vector_type(self) -> None:
        """Test that EmbeddingVector is a list of floats."""
        # This is a type alias, so we just verify it works as expected
        vector: EmbeddingVector = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert len(vector) == 5
        assert all(isinstance(v, float) for v in vector)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""
    
    def test_rag_error_base(self) -> None:
        """Test base RAGError exception."""
        error = RAGError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details is None
    
    def test_rag_error_with_details(self) -> None:
        """Test RAGError with details."""
        error = RAGError("Something went wrong", details="More info here")
        assert "Something went wrong" in str(error)
        assert "More info here" in str(error)
        assert error.details == "More info here"
    
    def test_connector_error_inherits_from_rag_error(self) -> None:
        """Test that ConnectorError inherits from RAGError."""
        error = ConnectorError("Connection failed")
        assert isinstance(error, RAGError)
        assert isinstance(error, Exception)
    
    def test_ingestion_error_inherits_from_rag_error(self) -> None:
        """Test that IngestionError inherits from RAGError."""
        error = IngestionError("Ingestion failed")
        assert isinstance(error, RAGError)
    
    def test_embedding_error_inherits_from_rag_error(self) -> None:
        """Test that EmbeddingError inherits from RAGError."""
        error = EmbeddingError("Embedding generation failed")
        assert isinstance(error, RAGError)
    
    def test_vector_store_error_inherits_from_rag_error(self) -> None:
        """Test that VectorStoreError inherits from RAGError."""
        error = VectorStoreError("Vector store operation failed")
        assert isinstance(error, RAGError)
    
    def test_retrieval_error_inherits_from_rag_error(self) -> None:
        """Test that RetrievalError inherits from RAGError."""
        error = RetrievalError("Retrieval failed")
        assert isinstance(error, RAGError)
    
    def test_llm_error_inherits_from_rag_error(self) -> None:
        """Test that LLMError inherits from RAGError."""
        error = LLMError("LLM operation failed")
        assert isinstance(error, RAGError)
    
    def test_configuration_error_inherits_from_rag_error(self) -> None:
        """Test that ConfigurationError inherits from RAGError."""
        error = ConfigurationError("Invalid configuration")
        assert isinstance(error, RAGError)
    
    def test_catch_all_rag_errors(self) -> None:
        """Test that all specific errors can be caught as RAGError."""
        errors = [
            ConnectorError("connector"),
            IngestionError("ingestion"),
            EmbeddingError("embedding"),
            VectorStoreError("vector_store"),
            RetrievalError("retrieval"),
            LLMError("llm"),
            ConfigurationError("configuration"),
        ]
        
        for error in errors:
            with pytest.raises(RAGError):
                raise error
    
    def test_exception_with_details_string_representation(self) -> None:
        """Test string representation of exceptions with details."""
        error = LLMError(
            "Failed to generate response",
            details="Context length exceeded (8192 tokens)"
        )
        error_str = str(error)
        assert "Failed to generate response" in error_str
        assert "Context length exceeded" in error_str
