"""AWS Bedrock embedding provider using Titan model.

This module provides an embedding implementation using AWS Bedrock's Titan
embedding model (amazon.titan-embed-text-v1), which produces 1536-dimensional
vectors.

The module handles AWS credentials from environment variables, implements
retry logic with exponential backoff, and handles rate limiting.
"""

import json
import logging
import os
import random
import time
from typing import Any

from src.embeddings.base import BaseEmbeddings
from src.exceptions import EmbeddingError
from src.types import EmbeddingVector

logger = logging.getLogger(__name__)

# Import boto3 at module level for easier mocking in tests
# The actual client is created lazily in _get_client()
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore
    BOTO3_AVAILABLE = False


class BedrockEmbeddings(BaseEmbeddings):
    """AWS Bedrock embedding provider using Titan model.
    
    This class provides embedding functionality using AWS Bedrock's Titan
    embedding model (amazon.titan-embed-text-v1). It handles AWS credentials
    from environment variables, implements retry logic with exponential
    backoff, and handles rate limiting.
    
    The Titan model produces 1536-dimensional vectors and is optimized for
    text similarity and semantic search applications.
    
    Attributes:
        model_id: The Bedrock model identifier.
        dimension: The dimensionality of the embedding vectors (1536).
        region: AWS region for the Bedrock service.
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries in seconds.
        max_batch_size: Maximum number of texts per batch request.
    
    Example:
        ```python
        # Basic usage with environment credentials
        embeddings = BedrockEmbeddings()
        vector = embeddings.embed_query("What is federalism?")
        print(f"Dimension: {embeddings.dimension}")  # 1536
        
        # With custom region
        embeddings = BedrockEmbeddings(region="us-west-2")
        
        # Embed multiple documents
        vectors = embeddings.embed_documents(["Hello", "World"])
        ```
    """
    
    # Titan embedding model produces 1536-dimensional vectors
    _DIMENSION = 1536
    _DEFAULT_MODEL_ID = "amazon.titan-embed-text-v1"
    _DEFAULT_REGION = "us-east-1"
    _DEFAULT_MAX_RETRIES = 5
    _DEFAULT_RETRY_DELAY = 1.0
    _DEFAULT_MAX_BATCH_SIZE = 1  # Bedrock Titan doesn't support batch
    
    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        max_batch_size: int | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        """Initialize the Bedrock embedding provider.
        
        Args:
            model_id: Bedrock model identifier. Defaults to amazon.titan-embed-text-v1.
            region: AWS region for Bedrock service. Defaults to us-east-1 or
                AWS_DEFAULT_REGION/AWS_REGION environment variable.
            max_retries: Maximum number of retry attempts for transient errors.
                Defaults to 5.
            retry_delay: Initial delay between retries in seconds. Defaults to 1.0.
                Uses exponential backoff with jitter.
            max_batch_size: Maximum number of texts per batch. Note: Titan
                doesn't support batch embedding, so this is used for parallel
                request management. Defaults to 1.
            aws_access_key_id: AWS access key ID. If not provided, uses
                AWS_ACCESS_KEY_ID environment variable or IAM role.
            aws_secret_access_key: AWS secret access key. If not provided,
                uses AWS_SECRET_ACCESS_KEY environment variable or IAM role.
            aws_session_token: AWS session token for temporary credentials.
            endpoint_url: Custom endpoint URL for testing or VPC endpoints.
        
        Raises:
            EmbeddingError: If AWS credentials are not configured properly.
        
        Example:
            ```python
            # Using environment credentials (recommended)
            embeddings = BedrockEmbeddings()
            
            # Using explicit credentials
            embeddings = BedrockEmbeddings(
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                region="us-west-2"
            )
            ```
        """
        self._model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", self._DEFAULT_MODEL_ID
        )
        self._region = region or os.environ.get(
            "AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", self._DEFAULT_REGION)
        )
        self._max_retries = max_retries or self._DEFAULT_MAX_RETRIES
        self._retry_delay = retry_delay or self._DEFAULT_RETRY_DELAY
        self._max_batch_size = max_batch_size or self._DEFAULT_MAX_BATCH_SIZE
        
        # Store credentials (will be used when creating client)
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._endpoint_url = endpoint_url
        
        # Lazy initialization of boto3 client
        self._client: Any = None
        self._client_initialized = False
        
        # Validate that credentials are available (deferred to first use)
        logger.debug(
            f"BedrockEmbeddings initialized with model_id={self._model_id}, "
            f"region={self._region}"
        )
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.
        
        Returns:
            The number of dimensions in each embedding vector (1536 for Titan).
        """
        return self._DIMENSION
    
    @property
    def model_id(self) -> str:
        """Return the Bedrock model identifier.
        
        Returns:
            The model identifier string.
        """
        return self._model_id
    
    @property
    def region(self) -> str:
        """Return the AWS region.
        
        Returns:
            The AWS region string.
        """
        return self._region
    
    @property
    def max_retries(self) -> int:
        """Return the maximum number of retry attempts.
        
        Returns:
            The maximum retry count.
        """
        return self._max_retries
    
    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size.
        
        Returns:
            The maximum number of texts per batch.
        """
        return self._max_batch_size
    
    def _get_client(self) -> Any:
        """Get or create the Bedrock boto3 client.
        
        This method lazily initializes the boto3 client on first use,
        allowing for proper error handling and testing.
        
        Returns:
            The boto3 Bedrock runtime client.
        
        Raises:
            EmbeddingError: If the client cannot be created.
        """
        if not self._client_initialized:
            if not BOTO3_AVAILABLE or boto3 is None:
                raise EmbeddingError(
                    "boto3 library not installed",
                    details="Install boto3 with: pip install boto3"
                )
            
            try:
                # Build client kwargs
                client_kwargs: dict[str, Any] = {
                    "service_name": "bedrock-runtime",
                    "region_name": self._region,
                }
                
                # Add credentials if provided
                if self._aws_access_key_id and self._aws_secret_access_key:
                    client_kwargs["aws_access_key_id"] = self._aws_access_key_id
                    client_kwargs["aws_secret_access_key"] = self._aws_secret_access_key
                    if self._aws_session_token:
                        client_kwargs["aws_session_token"] = self._aws_session_token
                
                # Add custom endpoint if provided
                if self._endpoint_url:
                    client_kwargs["endpoint_url"] = self._endpoint_url
                
                self._client = boto3.client(**client_kwargs)
                self._client_initialized = True
                logger.debug("Bedrock client initialized successfully")
                
            except Exception as e:
                raise EmbeddingError(
                    "Failed to initialize Bedrock client",
                    details=str(e)
                ) from e
        
        return self._client
    
    def _invoke_model(self, text: str) -> EmbeddingVector:
        """Invoke the Bedrock model to get embedding for a single text.
        
        This method handles the API call to Bedrock with retry logic
        and exponential backoff.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector for the text.
            
        Raises:
            EmbeddingError: If the API call fails after all retries.
        """
        client = self._get_client()
        
        # Prepare the request body for Titan model
        request_body = json.dumps({
            "inputText": text
        })
        
        last_exception: Exception | None = None
        
        for attempt in range(self._max_retries + 1):
            try:
                response = client.invoke_model(
                    modelId=self._model_id,
                    body=request_body,
                    accept="application/json",
                    contentType="application/json"
                )
                
                # Parse the response
                response_body = json.loads(response["body"].read())
                
                # Titan returns embedding in "embedding" field
                embedding = response_body["embedding"]
                
                return embedding
                
            except client.exceptions.ThrottlingException as e:
                # Rate limiting - always retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Rate limited by Bedrock, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "Bedrock rate limit exceeded",
                        details=f"Max retries ({self._max_retries}) exceeded"
                    ) from e
                    
            except client.exceptions.ServiceException as e:
                # Server-side error - retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Bedrock service error, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "Bedrock service error",
                        details=str(e)
                    ) from e
                    
            except client.exceptions.ValidationException as e:
                # Client error - don't retry
                raise EmbeddingError(
                    "Invalid request to Bedrock",
                    details=str(e)
                ) from e
                
            except client.exceptions.AccessDeniedException as e:
                # Authentication error - don't retry
                raise EmbeddingError(
                    "Access denied to Bedrock model",
                    details="Check AWS credentials and model access permissions"
                ) from e
                
            except Exception as e:
                # Unknown error - retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Unexpected error from Bedrock, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "Failed to get embedding from Bedrock",
                        details=str(e)
                    ) from e
        
        # This should never be reached, but satisfies type checker
        raise EmbeddingError(
            "Failed to get embedding from Bedrock",
            details=str(last_exception)
        )
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the backoff delay for a given retry attempt.
        
        Uses exponential backoff with jitter to avoid thundering herd.
        
        Args:
            attempt: The current attempt number (0-indexed).
            
        Returns:
            The delay in seconds before the next retry.
        """
        import random
        
        # Exponential backoff: delay * 2^attempt
        base_delay = self._retry_delay * (2 ** attempt)
        
        # Add jitter (random value between 0 and 1)
        jitter = random.random()
        
        # Cap at 60 seconds
        return min(base_delay + jitter, 60.0)
    
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single query into a vector representation.
        
        Args:
            text: The query text to embed.
            
        Returns:
            An embedding vector representing the query.
            
        Raises:
            EmbeddingError: If the embedding operation fails.
            ValueError: If text is empty or invalid.
        
        Example:
            ```python
            embeddings = BedrockEmbeddings()
            vector = embeddings.embed_query("What is federalism?")
            print(len(vector))  # 1536
            ```
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        return self._invoke_model(text.strip())
    
    def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a list of documents into vector representations.
        
        Since Bedrock Titan doesn't support batch embedding, this method
        makes individual calls for each text. It handles rate limiting
        with retries and backoff.
        
        Args:
            texts: A list of text strings to embed.
            
        Returns:
            A list of embedding vectors, one for each input text.
            The order of outputs matches the order of inputs.
            
        Raises:
            EmbeddingError: If the embedding operation fails.
            ValueError: If texts is empty or contains invalid values.
        
        Example:
            ```python
            embeddings = BedrockEmbeddings()
            texts = ["Hello world", "Goodbye world"]
            vectors = embeddings.embed_documents(texts)
            print(len(vectors))  # 2
            print(len(vectors[0]))  # 1536
            ```
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        
        # Process each text individually (Titan doesn't support batch)
        embeddings: list[EmbeddingVector] = []
        
        for i, text in enumerate(texts):
            logger.debug(f"Embedding document {i + 1}/{len(texts)}")
            embedding = self._invoke_model(text.strip())
            embeddings.append(embedding)
        
        return embeddings
    
    def __repr__(self) -> str:
        """Return string representation of the BedrockEmbeddings instance."""
        return (
            f"BedrockEmbeddings("
            f"model_id={self._model_id!r}, "
            f"region={self._region!r}, "
            f"dimension={self.dimension})"
        )