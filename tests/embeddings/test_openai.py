"""Tests for the OpenAIEmbeddings class.

This module tests the OpenAIEmbeddings implementation using OpenAI's
text-embedding-3 models. Tests use mocked OpenAI clients for unit tests
and skip integration tests if OPENAI_API_KEY is not available.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.embeddings.base import BaseEmbeddings
from src.exceptions import EmbeddingError


class TestOpenAIEmbeddingsBasic:
    """Basic tests for OpenAIEmbeddings that don't require API connection."""
    
    def test_import_openai_embeddings(self) -> None:
        """Test that OpenAIEmbeddings can be imported."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        assert OpenAIEmbeddings is not None
    
    def test_openai_embeddings_is_subclass_of_base(self) -> None:
        """Test that OpenAIEmbeddings is a subclass of BaseEmbeddings."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        assert issubclass(OpenAIEmbeddings, BaseEmbeddings)
    
    def test_default_model(self) -> None:
        """Test that the default model is text-embedding-3-small."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        assert embeddings.model == "text-embedding-3-small"
    
    def test_custom_model_small(self) -> None:
        """Test that text-embedding-3-small can be set."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        assert embeddings.model == "text-embedding-3-small"
    
    def test_custom_model_large(self) -> None:
        """Test that text-embedding-3-large can be set."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        assert embeddings.model == "text-embedding-3-large"
    
    def test_dimension_small_model(self) -> None:
        """Test that dimension is 1536 for small model."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        assert embeddings.dimension == 1536
    
    def test_dimension_large_model(self) -> None:
        """Test that dimension is 3072 for large model."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        assert embeddings.dimension == 3072
    
    def test_dimension_default_model(self) -> None:
        """Test that dimension is 1536 for default model."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        assert embeddings.dimension == 1536
    
    def test_default_max_retries(self) -> None:
        """Test that max_retries defaults to 5."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        assert embeddings.max_retries == 5
    
    def test_custom_max_retries(self) -> None:
        """Test that max_retries can be customized."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(max_retries=10)
        assert embeddings.max_retries == 10
    
    def test_default_max_batch_size(self) -> None:
        """Test that max_batch_size defaults to 100."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        assert embeddings.max_batch_size == 100
    
    def test_custom_max_batch_size(self) -> None:
        """Test that max_batch_size can be customized."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(max_batch_size=50)
        assert embeddings.max_batch_size == 50
    
    def test_repr_shows_model(self) -> None:
        """Test that __repr__ includes the model name."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        repr_str = repr(embeddings)
        
        assert "text-embedding-3-small" in repr_str
        assert "OpenAIEmbeddings" in repr_str
        assert "1536" in repr_str
    
    def test_model_from_environment(self) -> None:
        """Test that model can be set from environment variable."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Save original env
        original = os.environ.get("OPENAI_EMBEDDING_MODEL")
        
        try:
            os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-large"
            embeddings = OpenAIEmbeddings()
            assert embeddings.model == "text-embedding-3-large"
        finally:
            # Restore original env
            if original is not None:
                os.environ["OPENAI_EMBEDDING_MODEL"] = original
            else:
                os.environ.pop("OPENAI_EMBEDDING_MODEL", None)


class TestOpenAIEmbeddingsValidation:
    """Tests for input validation in OpenAIEmbeddings."""
    
    def test_embed_query_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_query("")
    
    def test_embed_query_whitespace_raises_error(self) -> None:
        """Test that whitespace-only string raises ValueError."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_query("   ")
    
    def test_embed_documents_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_documents([])
    
    def test_embed_documents_with_empty_string_raises_error(self) -> None:
        """Test that list with empty string raises ValueError."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_documents(["valid text", ""])
    
    def test_embed_documents_with_whitespace_raises_error(self) -> None:
        """Test that list with whitespace string raises ValueError."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_documents(["valid text", "   "])


class TestOpenAIEmbeddingsMocked:
    """Tests using mocked OpenAI client."""
    
    def _create_mock_embedding_data(self, index: int, embedding: list[float]) -> Mock:
        """Create a mock embedding data object.
        
        Args:
            index: The index of the embedding.
            embedding: The embedding vector.
            
        Returns:
            A mock embedding data object.
        """
        mock_data = Mock()
        mock_data.index = index
        mock_data.embedding = embedding
        return mock_data
    
    def _create_mock_response(self, embeddings: list[list[float]]) -> Mock:
        """Create a mock response from OpenAI API.
        
        Args:
            embeddings: List of embedding vectors to return.
            
        Returns:
            A mock response object.
        """
        mock_response = Mock()
        mock_response.data = [
            self._create_mock_embedding_data(i, emb) 
            for i, emb in enumerate(embeddings)
        ]
        return mock_response
    
    def _create_mock_client(self) -> Mock:
        """Create a mock OpenAI client.
        
        Returns:
            A mock OpenAI client.
        """
        mock_client = Mock()
        mock_client.embeddings = Mock()
        return mock_client
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_embed_single_text(self, mock_openai: Mock) -> None:
        """Test embedding a single text."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = self._create_mock_response([expected_embedding])
        mock_openai.return_value = mock_client
        
        # Test
        embeddings = OpenAIEmbeddings()
        result = embeddings.embed_query("What is federalism?")
        
        assert len(result) == 1536
        assert result == expected_embedding
        
        # Verify the call
        mock_client.embeddings.create.assert_called_once()
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == ["What is federalism?"]
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_embed_multiple_texts(self, mock_openai: Mock) -> None:
        """Test embedding multiple texts (batch)."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        embeddings_to_return = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536,
        ]
        mock_client.embeddings.create.return_value = self._create_mock_response(embeddings_to_return)
        mock_openai.return_value = mock_client
        
        # Test
        embeddings = OpenAIEmbeddings()
        texts = ["Hello world", "Goodbye world", "Test sentence"]
        results = embeddings.embed_documents(texts)
        
        assert len(results) == 3
        assert all(len(r) == 1536 for r in results)
        assert results[0] == embeddings_to_return[0]
        assert results[1] == embeddings_to_return[1]
        assert results[2] == embeddings_to_return[2]
        
        # Verify the call
        mock_client.embeddings.create.assert_called_once()
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_embed_query_strips_whitespace(self, mock_openai: Mock) -> None:
        """Test that embed_query strips whitespace from text."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = self._create_mock_response([expected_embedding])
        mock_openai.return_value = mock_client
        
        # Test
        embeddings = OpenAIEmbeddings()
        embeddings.embed_query("  hello world  ")
        
        # Verify text was stripped
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == ["hello world"]
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_embed_documents_strips_whitespace(self, mock_openai: Mock) -> None:
        """Test that embed_documents strips whitespace from texts."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        embeddings_to_return = [[0.1] * 1536, [0.2] * 1536]
        mock_client.embeddings.create.return_value = self._create_mock_response(embeddings_to_return)
        mock_openai.return_value = mock_client
        
        # Test
        embeddings = OpenAIEmbeddings()
        embeddings.embed_documents(["  hello  ", "  world  "])
        
        # Verify texts were stripped
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == ["hello", "world"]
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_batch_processing(self, mock_openai: Mock) -> None:
        """Test that large lists are batched correctly."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        
        # Create 150 texts - should be split into 2 batches (100 + 50)
        texts = [f"Text {i}" for i in range(150)]
        
        # First batch returns 100 embeddings, second returns 50
        first_batch = [[0.1] * 1536 for _ in range(100)]
        second_batch = [[0.2] * 1536 for _ in range(50)]
        
        mock_client.embeddings.create.side_effect = [
            self._create_mock_response(first_batch),
            self._create_mock_response(second_batch),
        ]
        mock_openai.return_value = mock_client
        
        # Test with batch size 100
        embeddings = OpenAIEmbeddings(max_batch_size=100)
        results = embeddings.embed_documents(texts)
        
        assert len(results) == 150
        assert mock_client.embeddings.create.call_count == 2
        
        # Verify first batch
        first_call_kwargs = mock_client.embeddings.create.call_args_list[0].kwargs
        assert len(first_call_kwargs["input"]) == 100
        
        # Verify second batch
        second_call_kwargs = mock_client.embeddings.create.call_args_list[1].kwargs
        assert len(second_call_kwargs["input"]) == 50
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_retry_on_rate_limit(self, mock_openai: Mock) -> None:
        """Test retry logic on rate limit error."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        from openai import RateLimitError
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        
        # First call raises rate limit, second succeeds
        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            self._create_mock_response([expected_embedding]),
        ]
        mock_openai.return_value = mock_client
        
        # Test with patched sleep to speed up test
        with patch("src.embeddings.openai_emb.time.sleep"):
            embeddings = OpenAIEmbeddings(max_retries=3)
            result = embeddings.embed_query("test query")
        
        assert result == expected_embedding
        assert mock_client.embeddings.create.call_count == 2
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_retry_on_timeout(self, mock_openai: Mock) -> None:
        """Test retry logic on timeout error."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        from openai import APITimeoutError
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        
        # First call raises timeout, second succeeds
        mock_client.embeddings.create.side_effect = [
            APITimeoutError("Request timed out"),
            self._create_mock_response([expected_embedding]),
        ]
        mock_openai.return_value = mock_client
        
        # Test with patched sleep
        with patch("src.embeddings.openai_emb.time.sleep"):
            embeddings = OpenAIEmbeddings(max_retries=3)
            result = embeddings.embed_query("test query")
        
        assert result == expected_embedding
        assert mock_client.embeddings.create.call_count == 2
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_retry_on_connection_error(self, mock_openai: Mock) -> None:
        """Test retry logic on connection error."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        from openai import APIConnectionError
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        
        # First call raises connection error, second succeeds
        # APIConnectionError takes request= as positional argument
        mock_request = Mock()
        mock_client.embeddings.create.side_effect = [
            APIConnectionError(request=mock_request),
            self._create_mock_response([expected_embedding]),
        ]
        mock_openai.return_value = mock_client
        
        # Test with patched sleep
        with patch("src.embeddings.openai_emb.time.sleep"):
            embeddings = OpenAIEmbeddings(max_retries=3)
            result = embeddings.embed_query("test query")
        
        assert result == expected_embedding
        assert mock_client.embeddings.create.call_count == 2
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_max_retries_exceeded(self, mock_openai: Mock) -> None:
        """Test that error is raised after max retries exceeded."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        from openai import RateLimitError
        
        # Setup mock - always raise rate limit
        mock_client = self._create_mock_client()
        mock_client.embeddings.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(), body=None
        )
        mock_openai.return_value = mock_client
        
        # Test with patched sleep
        with patch("src.embeddings.openai_emb.time.sleep"):
            embeddings = OpenAIEmbeddings(max_retries=2)
            
            with pytest.raises(EmbeddingError, match="rate limit exceeded"):
                embeddings.embed_query("test query")
        
        # Should have tried max_retries + 1 times (initial + retries)
        assert mock_client.embeddings.create.call_count == 3
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_api_error_no_retry_for_client_error(self, mock_openai: Mock) -> None:
        """Test that client errors (4xx) do not retry."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        from openai import APIError
        
        # Setup mock
        mock_client = self._create_mock_client()
        
        # Create a mock error with 400 status code
        # APIError takes message, request, and body as positional args
        mock_request = Mock()
        mock_client.embeddings.create.side_effect = APIError(
            "Bad request",
            request=mock_request,
            body=None
        )
        # Set status_code attribute after creation
        mock_client.embeddings.create.side_effect.status_code = 400
        mock_openai.return_value = mock_client
        
        # Test
        embeddings = OpenAIEmbeddings(max_retries=5)
        
        with pytest.raises(EmbeddingError, match="API error"):
            embeddings.embed_query("test query")
        
        # Should only have tried once
        assert mock_client.embeddings.create.call_count == 1
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_api_error_retry_for_server_error(self, mock_openai: Mock) -> None:
        """Test that server errors (5xx) do retry."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        from openai import APIError
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        
        # Create a mock error with 500 status code
        # APIError takes message, request, and body as positional args
        mock_request = Mock()
        server_error = APIError("Internal server error", request=mock_request, body=None)
        server_error.status_code = 500
        
        # First call raises server error, second succeeds
        mock_client.embeddings.create.side_effect = [
            server_error,
            self._create_mock_response([expected_embedding]),
        ]
        mock_openai.return_value = mock_client
        
        # Test with patched sleep
        with patch("src.embeddings.openai_emb.time.sleep"):
            embeddings = OpenAIEmbeddings(max_retries=3)
            result = embeddings.embed_query("test query")
        
        assert result == expected_embedding
        assert mock_client.embeddings.create.call_count == 2
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_custom_api_key(self, mock_openai: Mock) -> None:
        """Test that custom API key is passed to OpenAI client."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = self._create_mock_response([expected_embedding])
        mock_openai.return_value = mock_client
        
        # Test with custom API key
        embeddings = OpenAIEmbeddings(api_key="sk-test-key")
        embeddings.embed_query("test query")
        
        # Verify API key was passed
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test-key"
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_custom_base_url(self, mock_openai: Mock) -> None:
        """Test that custom base URL is passed to OpenAI client."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = self._create_mock_response([expected_embedding])
        mock_openai.return_value = mock_client
        
        # Test with custom base URL
        embeddings = OpenAIEmbeddings(base_url="https://custom.api.url/v1")
        embeddings.embed_query("test query")
        
        # Verify base URL was passed
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api.url/v1"
    
    @patch("src.embeddings.openai_emb.OpenAI")
    def test_large_model_embedding(self, mock_openai: Mock) -> None:
        """Test embedding with large model (3072 dimensions)."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 3072
        mock_client.embeddings.create.return_value = self._create_mock_response([expected_embedding])
        mock_openai.return_value = mock_client
        
        # Test with large model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        result = embeddings.embed_query("test query")
        
        assert len(result) == 3072
        
        # Verify model was used
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-large"


class TestOpenAIEmbeddingsIntegration:
    """Integration tests that require OPENAI_API_KEY."""
    
    @pytest.fixture
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.environ.get("OPENAI_API_KEY")
    
    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not set in environment"
    )
    def test_real_embedding_single_text(self, openai_api_key: str | None) -> None:
        """Test embedding a single text with real API."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        result = embeddings.embed_query("What is federalism?")
        
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not set in environment"
    )
    def test_real_embedding_multiple_texts(self, openai_api_key: str | None) -> None:
        """Test embedding multiple texts with real API."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        texts = ["Hello world", "Goodbye world"]
        results = embeddings.embed_documents(texts)
        
        assert len(results) == 2
        assert all(len(r) == 1536 for r in results)
    
    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not set in environment"
    )
    def test_real_embedding_large_model(self, openai_api_key: str | None) -> None:
        """Test embedding with large model using real API."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=openai_api_key
        )
        result = embeddings.embed_query("What is federalism?")
        
        assert len(result) == 3072
        assert all(isinstance(x, float) for x in result)


class TestOpenAIEmbeddingsBackoff:
    """Tests for backoff calculation."""
    
    def test_backoff_delay_increases(self) -> None:
        """Test that backoff delay increases with attempts."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(retry_delay=1.0)
        
        # Test that delays increase (approximately)
        delay_0 = embeddings._calculate_backoff_delay(0)
        delay_1 = embeddings._calculate_backoff_delay(1)
        delay_2 = embeddings._calculate_backoff_delay(2)
        
        # Allow for jitter, but base should increase
        assert delay_0 < delay_1 + 1  # Allow for jitter variance
        assert delay_1 < delay_2 + 1
    
    def test_backoff_delay_capped_at_60(self) -> None:
        """Test that backoff delay is capped at 60 seconds."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(retry_delay=1.0)
        
        # Very high attempt number
        delay = embeddings._calculate_backoff_delay(10)
        
        assert delay <= 60.0
    
    def test_backoff_includes_jitter(self) -> None:
        """Test that backoff delay includes jitter."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(retry_delay=1.0)
        
        # Multiple calls should produce slightly different results due to jitter
        delays = [embeddings._calculate_backoff_delay(0) for _ in range(10)]
        
        # Not all delays should be exactly the same
        # (This test could theoretically fail if all random values are the same,
        # but that's extremely unlikely)
        unique_delays = set(delays)
        assert len(unique_delays) > 1


class TestOpenAIEmbeddingsLibraryNotInstalled:
    """Tests for handling when openai library is not installed."""
    
    def test_error_when_openai_not_installed(self) -> None:
        """Test that helpful error is raised when openai is not installed."""
        from src.embeddings.openai_emb import OpenAIEmbeddings
        
        # Patch OPENAI_AVAILABLE to simulate library not being installed
        with patch("src.embeddings.openai_emb.OPENAI_AVAILABLE", False):
            embeddings = OpenAIEmbeddings()
            
            with pytest.raises(EmbeddingError, match="openai library not installed"):
                embeddings.embed_query("test query")
