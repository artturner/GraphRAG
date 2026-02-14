"""Tests for the BedrockEmbeddings class.

This module tests the BedrockEmbeddings implementation using AWS Bedrock
Titan model. Tests use mocked boto3 clients for unit tests and skip
integration tests if AWS credentials are not available.
"""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.embeddings.base import BaseEmbeddings
from src.exceptions import EmbeddingError


class TestBedrockEmbeddingsBasic:
    """Basic tests for BedrockEmbeddings that don't require AWS connection."""
    
    def test_import_bedrock_embeddings(self) -> None:
        """Test that BedrockEmbeddings can be imported."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        assert BedrockEmbeddings is not None
    
    def test_bedrock_embeddings_is_subclass_of_base(self) -> None:
        """Test that BedrockEmbeddings is a subclass of BaseEmbeddings."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        assert issubclass(BedrockEmbeddings, BaseEmbeddings)
    
    def test_default_model_id(self) -> None:
        """Test that the default model ID is amazon.titan-embed-text-v1."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        assert embeddings.model_id == "amazon.titan-embed-text-v1"
    
    def test_custom_model_id(self) -> None:
        """Test that a custom model ID can be set."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings(model_id="custom-model-id")
        assert embeddings.model_id == "custom-model-id"
    
    def test_default_region(self) -> None:
        """Test that the default region is us-east-1."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        assert embeddings.region == "us-east-1"
    
    def test_custom_region(self) -> None:
        """Test that a custom region can be set."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings(region="us-west-2")
        assert embeddings.region == "us-west-2"
    
    def test_region_from_environment(self) -> None:
        """Test that region can be set from environment variable."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Save original env
        original = os.environ.get("AWS_DEFAULT_REGION")
        
        try:
            os.environ["AWS_DEFAULT_REGION"] = "eu-west-1"
            embeddings = BedrockEmbeddings()
            assert embeddings.region == "eu-west-1"
        finally:
            # Restore original env
            if original is not None:
                os.environ["AWS_DEFAULT_REGION"] = original
            else:
                os.environ.pop("AWS_DEFAULT_REGION", None)
    
    def test_dimension_is_1536(self) -> None:
        """Test that dimension is 1536 for Titan model."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        assert embeddings.dimension == 1536
    
    def test_default_max_retries(self) -> None:
        """Test that max_retries defaults to 5."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        assert embeddings.max_retries == 5
    
    def test_custom_max_retries(self) -> None:
        """Test that max_retries can be customized."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings(max_retries=10)
        assert embeddings.max_retries == 10
    
    def test_default_max_batch_size(self) -> None:
        """Test that max_batch_size defaults to 1."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        assert embeddings.max_batch_size == 1
    
    def test_repr_shows_model_id(self) -> None:
        """Test that __repr__ includes the model ID."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        repr_str = repr(embeddings)
        
        assert "amazon.titan-embed-text-v1" in repr_str
        assert "BedrockEmbeddings" in repr_str
        assert "1536" in repr_str


class TestBedrockEmbeddingsValidation:
    """Tests for input validation in BedrockEmbeddings."""
    
    def test_embed_query_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_query("")
    
    def test_embed_query_whitespace_raises_error(self) -> None:
        """Test that whitespace-only string raises ValueError."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_query("   ")
    
    def test_embed_documents_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_documents([])
    
    def test_embed_documents_with_empty_string_raises_error(self) -> None:
        """Test that list with empty string raises ValueError."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_documents(["valid text", ""])
    
    def test_embed_documents_with_whitespace_raises_error(self) -> None:
        """Test that list with whitespace string raises ValueError."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embeddings.embed_documents(["valid text", "   "])


class TestBedrockEmbeddingsMocked:
    """Tests using mocked boto3 client."""
    
    def _create_mock_response(self, embedding: list[float]) -> Mock:
        """Create a mock response from Bedrock API.
        
        Args:
            embedding: The embedding vector to return.
            
        Returns:
            A mock response object.
        """
        mock_body = Mock()
        mock_body.read.return_value = json.dumps({"embedding": embedding})
        
        return {"body": mock_body}
    
    def _create_mock_client(self) -> Mock:
        """Create a mock Bedrock client.
        
        Returns:
            A mock boto3 Bedrock client.
        """
        mock_client = Mock()
        mock_client.exceptions = Mock()
        mock_client.exceptions.ThrottlingException = type("ThrottlingException", (Exception,), {})
        mock_client.exceptions.ServiceException = type("ServiceException", (Exception,), {})
        mock_client.exceptions.ValidationException = type("ValidationException", (Exception,), {})
        mock_client.exceptions.AccessDeniedException = type("AccessDeniedException", (Exception,), {})
        return mock_client
    
    @patch("src.embeddings.bedrock.boto3")
    def test_embed_single_text(self, mock_boto3: Mock) -> None:
        """Test embedding a single text."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.invoke_model.return_value = self._create_mock_response(expected_embedding)
        mock_boto3.client.return_value = mock_client
        
        # Test
        embeddings = BedrockEmbeddings()
        result = embeddings.embed_query("What is federalism?")
        
        assert len(result) == 1536
        assert result == expected_embedding
        
        # Verify the call
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args
        assert call_args.kwargs["modelId"] == "amazon.titan-embed-text-v1"
        
        # Verify request body
        body = json.loads(call_args.kwargs["body"])
        assert body["inputText"] == "What is federalism?"
    
    @patch("src.embeddings.bedrock.boto3")
    def test_embed_multiple_texts(self, mock_boto3: Mock) -> None:
        """Test embedding multiple texts (batch)."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        embeddings_to_return = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536,
        ]
        mock_client.invoke_model.side_effect = [
            self._create_mock_response(e) for e in embeddings_to_return
        ]
        mock_boto3.client.return_value = mock_client
        
        # Test
        embeddings = BedrockEmbeddings()
        texts = ["Hello world", "Goodbye world", "Test sentence"]
        results = embeddings.embed_documents(texts)
        
        assert len(results) == 3
        assert all(len(r) == 1536 for r in results)
        assert results[0] == embeddings_to_return[0]
        assert results[1] == embeddings_to_return[1]
        assert results[2] == embeddings_to_return[2]
        
        # Verify all calls were made
        assert mock_client.invoke_model.call_count == 3
    
    @patch("src.embeddings.bedrock.boto3")
    def test_embed_query_strips_whitespace(self, mock_boto3: Mock) -> None:
        """Test that embed_query strips whitespace from text."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.invoke_model.return_value = self._create_mock_response(expected_embedding)
        mock_boto3.client.return_value = mock_client
        
        # Test
        embeddings = BedrockEmbeddings()
        embeddings.embed_query("  hello world  ")
        
        # Verify text was stripped
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["inputText"] == "hello world"
    
    @patch("src.embeddings.bedrock.boto3")
    def test_retry_on_throttling(self, mock_boto3: Mock) -> None:
        """Test retry logic on throttling exception."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        
        # First call raises throttling, second succeeds
        mock_client.invoke_model.side_effect = [
            mock_client.exceptions.ThrottlingException("Rate exceeded"),
            self._create_mock_response(expected_embedding),
        ]
        mock_boto3.client.return_value = mock_client
        
        # Test with patched sleep to speed up test
        with patch("src.embeddings.bedrock.time.sleep"):
            embeddings = BedrockEmbeddings(max_retries=3)
            result = embeddings.embed_query("test query")
        
        assert result == expected_embedding
        assert mock_client.invoke_model.call_count == 2
    
    @patch("src.embeddings.bedrock.boto3")
    def test_retry_on_service_exception(self, mock_boto3: Mock) -> None:
        """Test retry logic on service exception."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        
        # First call raises service exception, second succeeds
        mock_client.invoke_model.side_effect = [
            mock_client.exceptions.ServiceException("Internal error"),
            self._create_mock_response(expected_embedding),
        ]
        mock_boto3.client.return_value = mock_client
        
        # Test with patched sleep
        with patch("src.embeddings.bedrock.time.sleep"):
            embeddings = BedrockEmbeddings(max_retries=3)
            result = embeddings.embed_query("test query")
        
        assert result == expected_embedding
        assert mock_client.invoke_model.call_count == 2
    
    @patch("src.embeddings.bedrock.boto3")
    def test_max_retries_exceeded(self, mock_boto3: Mock) -> None:
        """Test that error is raised after max retries exceeded."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock - always raise throttling
        mock_client = self._create_mock_client()
        mock_client.invoke_model.side_effect = mock_client.exceptions.ThrottlingException(
            "Rate exceeded"
        )
        mock_boto3.client.return_value = mock_client
        
        # Test with patched sleep
        with patch("src.embeddings.bedrock.time.sleep"):
            embeddings = BedrockEmbeddings(max_retries=2)
            
            with pytest.raises(EmbeddingError, match="rate limit exceeded"):
                embeddings.embed_query("test query")
        
        # Should have tried max_retries + 1 times (initial + retries)
        assert mock_client.invoke_model.call_count == 3
    
    @patch("src.embeddings.bedrock.boto3")
    def test_validation_exception_no_retry(self, mock_boto3: Mock) -> None:
        """Test that validation exception does not retry."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        mock_client.invoke_model.side_effect = mock_client.exceptions.ValidationException(
            "Invalid input"
        )
        mock_boto3.client.return_value = mock_client
        
        # Test
        embeddings = BedrockEmbeddings(max_retries=5)
        
        with pytest.raises(EmbeddingError, match="Invalid request"):
            embeddings.embed_query("test query")
        
        # Should only have tried once
        assert mock_client.invoke_model.call_count == 1
    
    @patch("src.embeddings.bedrock.boto3")
    def test_access_denied_exception_no_retry(self, mock_boto3: Mock) -> None:
        """Test that access denied exception does not retry."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        mock_client.invoke_model.side_effect = mock_client.exceptions.AccessDeniedException(
            "Access denied"
        )
        mock_boto3.client.return_value = mock_client
        
        # Test
        embeddings = BedrockEmbeddings(max_retries=5)
        
        with pytest.raises(EmbeddingError, match="Access denied"):
            embeddings.embed_query("test query")
        
        # Should only have tried once
        assert mock_client.invoke_model.call_count == 1
    
    @patch("src.embeddings.bedrock.boto3")
    def test_custom_credentials(self, mock_boto3: Mock) -> None:
        """Test that custom credentials are passed to boto3."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.invoke_model.return_value = self._create_mock_response(expected_embedding)
        mock_boto3.client.return_value = mock_client
        
        # Test with custom credentials
        embeddings = BedrockEmbeddings(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-west-2"
        )
        embeddings.embed_query("test query")
        
        # Verify credentials were passed
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args.kwargs
        assert call_kwargs["aws_access_key_id"] == "AKIAIOSFODNN7EXAMPLE"
        assert call_kwargs["aws_secret_access_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert call_kwargs["region_name"] == "us-west-2"
    
    @patch("src.embeddings.bedrock.boto3")
    def test_custom_endpoint_url(self, mock_boto3: Mock) -> None:
        """Test that custom endpoint URL is passed to boto3."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Setup mock
        mock_client = self._create_mock_client()
        expected_embedding = [0.1] * 1536
        mock_client.invoke_model.return_value = self._create_mock_response(expected_embedding)
        mock_boto3.client.return_value = mock_client
        
        # Test with custom endpoint
        embeddings = BedrockEmbeddings(
            endpoint_url="https://bedrock-runtime.vpce-123.us-east-1.vpce.amazonaws.com"
        )
        embeddings.embed_query("test query")
        
        # Verify endpoint was passed
        call_kwargs = mock_boto3.client.call_args.kwargs
        assert call_kwargs["endpoint_url"] == "https://bedrock-runtime.vpce-123.us-east-1.vpce.amazonaws.com"


class TestBackoffCalculation:
    """Tests for exponential backoff calculation."""
    
    def test_backoff_increases_exponentially(self) -> None:
        """Test that backoff delay increases exponentially."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings(retry_delay=1.0)
        
        # Calculate delays for multiple attempts
        # Note: there's jitter, so we check the base value
        delays = []
        for attempt in range(5):
            # Remove jitter by mocking random
            with patch("src.embeddings.bedrock.random.random", return_value=0):
                delay = embeddings._calculate_backoff_delay(attempt)
                delays.append(delay)
        
        # Check exponential growth (without jitter)
        assert delays[0] == 1.0  # 1.0 * 2^0 = 1
        assert delays[1] == 2.0  # 1.0 * 2^1 = 2
        assert delays[2] == 4.0  # 1.0 * 2^2 = 4
        assert delays[3] == 8.0  # 1.0 * 2^3 = 8
        assert delays[4] == 16.0  # 1.0 * 2^4 = 16
    
    def test_backoff_capped_at_60_seconds(self) -> None:
        """Test that backoff delay is capped at 60 seconds."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings(retry_delay=1.0)
        
        # High attempt number would exceed 60s without cap
        with patch("src.embeddings.bedrock.random.random", return_value=0):
            delay = embeddings._calculate_backoff_delay(10)  # 1.0 * 2^10 = 1024
        
        assert delay == 60.0
    
    def test_backoff_includes_jitter(self) -> None:
        """Test that backoff delay includes random jitter."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings(retry_delay=1.0)
        
        # With jitter of 0.5
        with patch("src.embeddings.bedrock.random.random", return_value=0.5):
            delay = embeddings._calculate_backoff_delay(0)
        
        # Base 1.0 + jitter 0.5 = 1.5
        assert delay == 1.5


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""
    
    def test_model_id_from_environment(self) -> None:
        """Test that model ID can be set from environment variable."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        original = os.environ.get("BEDROCK_MODEL_ID")
        
        try:
            os.environ["BEDROCK_MODEL_ID"] = "custom-model"
            embeddings = BedrockEmbeddings()
            assert embeddings.model_id == "custom-model"
        finally:
            if original is not None:
                os.environ["BEDROCK_MODEL_ID"] = original
            else:
                os.environ.pop("BEDROCK_MODEL_ID", None)
    
    def test_aws_region_from_environment(self) -> None:
        """Test that AWS region can be set from AWS_REGION env var."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        # Clear AWS_DEFAULT_REGION if set
        original_default = os.environ.get("AWS_DEFAULT_REGION")
        original_region = os.environ.get("AWS_REGION")
        
        try:
            if "AWS_DEFAULT_REGION" in os.environ:
                del os.environ["AWS_DEFAULT_REGION"]
            os.environ["AWS_REGION"] = "ap-southeast-1"
            
            embeddings = BedrockEmbeddings()
            assert embeddings.region == "ap-southeast-1"
        finally:
            # Restore original values
            if original_default is not None:
                os.environ["AWS_DEFAULT_REGION"] = original_default
            elif "AWS_DEFAULT_REGION" in os.environ:
                del os.environ["AWS_DEFAULT_REGION"]
            
            if original_region is not None:
                os.environ["AWS_REGION"] = original_region
            elif "AWS_REGION" in os.environ:
                del os.environ["AWS_REGION"]


# Integration tests that require real AWS credentials
# These are skipped if credentials are not available
@pytest.mark.integration
class TestBedrockEmbeddingsIntegration:
    """Integration tests with real AWS Bedrock.
    
    These tests are skipped if AWS credentials are not available.
    Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
    to run these tests.
    """
    
    @pytest.fixture(scope="class")
    def aws_credentials_available(self) -> bool:
        """Check if AWS credentials are available."""
        return (
            os.environ.get("AWS_ACCESS_KEY_ID") is not None
            and os.environ.get("AWS_SECRET_ACCESS_KEY") is not None
        )
    
    @pytest.mark.skipif(
        not (
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        ),
        reason="AWS credentials not available"
    )
    def test_real_embed_query(self) -> None:
        """Test embedding a real query with Bedrock."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        vector = embeddings.embed_query("What is federalism?")
        
        assert len(vector) == 1536
        assert all(isinstance(v, float) for v in vector)
    
    @pytest.mark.skipif(
        not (
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        ),
        reason="AWS credentials not available"
    )
    def test_real_embed_documents(self) -> None:
        """Test embedding real documents with Bedrock."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        texts = [
            "Federalism is a system of government.",
            "The constitution divides power between federal and state governments.",
        ]
        vectors = embeddings.embed_documents(texts)
        
        assert len(vectors) == 2
        assert all(len(v) == 1536 for v in vectors)
    
    @pytest.mark.skipif(
        not (
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        ),
        reason="AWS credentials not available"
    )
    def test_real_dimension_property(self) -> None:
        """Test that dimension property matches actual embedding dimension."""
        from src.embeddings.bedrock import BedrockEmbeddings
        
        embeddings = BedrockEmbeddings()
        vector = embeddings.embed_query("test")
        
        assert len(vector) == embeddings.dimension