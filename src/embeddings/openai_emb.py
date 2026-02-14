"""OpenAI embedding provider using text-embedding-3 models.

This module provides an embedding implementation using OpenAI's text-embedding-3
models (text-embedding-3-small and text-embedding-3-large), which produce
1536-dimensional and 3072-dimensional vectors respectively.

The module handles API key from environment variables, implements retry logic
with exponential backoff, and handles rate limiting.
"""

import logging
import os
import random
import time
from typing import Any

from src.embeddings.base import BaseEmbeddings
from src.exceptions import EmbeddingError
from src.types import EmbeddingVector

logger = logging.getLogger(__name__)

# Import openai at module level for easier mocking in tests
# The actual client is created lazily in _get_client()
try:
    from openai import OpenAI, APIConnectionError, APIError, RateLimitError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    APIConnectionError = None  # type: ignore
    APIError = None  # type: ignore
    RateLimitError = None  # type: ignore
    APITimeoutError = None  # type: ignore
    OPENAI_AVAILABLE = False


# Model dimensions mapping
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embedding provider using text-embedding-3 models.
    
    This class provides embedding functionality using OpenAI's text-embedding-3
    models. It handles API key from environment variables, implements retry logic
    with exponential backoff, and handles rate limiting.
    
    The text-embedding-3-small model produces 1536-dimensional vectors, while
    text-embedding-3-large produces 3072-dimensional vectors.
    
    Attributes:
        model: The OpenAI model identifier.
        dimension: The dimensionality of the embedding vectors.
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries in seconds.
        max_batch_size: Maximum number of texts per batch request.
    
    Example:
        ```python
        # Basic usage with environment credentials
        embeddings = OpenAIEmbeddings()
        vector = embeddings.embed_query("What is federalism?")
        print(f"Dimension: {embeddings.dimension}")  # 1536
        
        # With large model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        print(f"Dimension: {embeddings.dimension}")  # 3072
        
        # Embed multiple documents
        vectors = embeddings.embed_documents(["Hello", "World"])
        ```
    """
    
    _DEFAULT_MODEL = "text-embedding-3-small"
    _DEFAULT_MAX_RETRIES = 5
    _DEFAULT_RETRY_DELAY = 1.0
    _DEFAULT_MAX_BATCH_SIZE = 100  # OpenAI supports batch embedding
    
    def __init__(
        self,
        model: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        max_batch_size: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI embedding provider.
        
        Args:
            model: OpenAI model identifier. Defaults to text-embedding-3-small.
                Options: text-embedding-3-small (1536d), text-embedding-3-large (3072d).
            max_retries: Maximum number of retry attempts for transient errors.
                Defaults to 5.
            retry_delay: Initial delay between retries in seconds. Defaults to 1.0.
                Uses exponential backoff with jitter.
            max_batch_size: Maximum number of texts per batch. OpenAI supports
                batch embedding up to 100 texts per request. Defaults to 100.
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY
                environment variable.
            base_url: Custom base URL for API endpoint (for testing or proxies).
        
        Raises:
            EmbeddingError: If OpenAI API key is not configured properly.
        
        Example:
            ```python
            # Using environment credentials (recommended)
            embeddings = OpenAIEmbeddings()
            
            # Using explicit API key
            embeddings = OpenAIEmbeddings(
                api_key="sk-...",
                model="text-embedding-3-large"
            )
            ```
        """
        self._model = model or os.environ.get(
            "OPENAI_EMBEDDING_MODEL", self._DEFAULT_MODEL
        )
        self._max_retries = max_retries or self._DEFAULT_MAX_RETRIES
        self._retry_delay = retry_delay or self._DEFAULT_RETRY_DELAY
        self._max_batch_size = max_batch_size or self._DEFAULT_MAX_BATCH_SIZE
        
        # Store credentials (will be used when creating client)
        self._api_key = api_key
        self._base_url = base_url
        
        # Lazy initialization of OpenAI client
        self._client: Any = None
        self._client_initialized = False
        
        # Validate model
        if self._model not in MODEL_DIMENSIONS:
            logger.warning(
                f"Unknown model '{self._model}', dimension will be determined at runtime. "
                f"Known models: {list(MODEL_DIMENSIONS.keys())}"
            )
        
        logger.debug(
            f"OpenAIEmbeddings initialized with model={self._model}"
        )
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.
        
        Returns:
            The number of dimensions in each embedding vector.
            1536 for text-embedding-3-small, 3072 for text-embedding-3-large.
        """
        return MODEL_DIMENSIONS.get(self._model, 1536)
    
    @property
    def model(self) -> str:
        """Return the OpenAI model identifier.
        
        Returns:
            The model identifier string.
        """
        return self._model
    
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
        """Get or create the OpenAI client.
        
        This method lazily initializes the OpenAI client on first use,
        allowing for proper error handling and testing.
        
        Returns:
            The OpenAI client instance.
        
        Raises:
            EmbeddingError: If the client cannot be created.
        """
        if not self._client_initialized:
            if not OPENAI_AVAILABLE or OpenAI is None:
                raise EmbeddingError(
                    "openai library not installed",
                    details="Install openai with: pip install openai"
                )
            
            try:
                # Build client kwargs
                client_kwargs: dict[str, Any] = {}
                
                # Add API key if provided
                api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
                if api_key:
                    client_kwargs["api_key"] = api_key
                
                # Add custom base URL if provided
                if self._base_url:
                    client_kwargs["base_url"] = self._base_url
                
                self._client = OpenAI(**client_kwargs)
                self._client_initialized = True
                logger.debug("OpenAI client initialized successfully")
                
            except Exception as e:
                raise EmbeddingError(
                    "Failed to initialize OpenAI client",
                    details=str(e)
                ) from e
        
        return self._client
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the backoff delay for a given retry attempt.
        
        Uses exponential backoff with jitter to avoid thundering herd.
        
        Args:
            attempt: The current attempt number (0-indexed).
            
        Returns:
            The delay in seconds before the next retry.
        """
        # Exponential backoff: delay * 2^attempt
        base_delay = self._retry_delay * (2 ** attempt)
        
        # Add jitter (random value between 0 and 1)
        jitter = random.random()
        
        # Cap at 60 seconds
        return min(base_delay + jitter, 60.0)
    
    def _create_embedding(self, texts: list[str]) -> list[EmbeddingVector]:
        """Create embeddings for a list of texts using OpenAI API.
        
        This method handles the API call to OpenAI with retry logic
        and exponential backoff.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If the API call fails after all retries.
        """
        client = self._get_client()
        
        last_exception: Exception | None = None
        
        for attempt in range(self._max_retries + 1):
            try:
                response = client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                
                # Extract embeddings in order
                embeddings = [
                    item.embedding for item in sorted(response.data, key=lambda x: x.index)
                ]
                
                return embeddings
                
            except RateLimitError as e:
                # Rate limiting - always retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Rate limited by OpenAI, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "OpenAI rate limit exceeded",
                        details=f"Max retries ({self._max_retries}) exceeded"
                    ) from e
            
            except APITimeoutError as e:
                # Timeout - retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"OpenAI API timeout, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "OpenAI API timeout",
                        details=str(e)
                    ) from e
            
            except APIConnectionError as e:
                # Connection error - retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"OpenAI API connection error, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "Failed to connect to OpenAI API",
                        details=str(e)
                    ) from e
            
            except APIError as e:
                # Server-side error - retry with backoff for 5xx errors
                last_exception = e
                if hasattr(e, 'status_code') and e.status_code and e.status_code >= 500:
                    if attempt < self._max_retries:
                        delay = self._calculate_backoff_delay(attempt)
                        logger.warning(
                            f"OpenAI API server error ({e.status_code}), retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{self._max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise EmbeddingError(
                            "OpenAI API server error",
                            details=str(e)
                        ) from e
                else:
                    # Client error - don't retry
                    raise EmbeddingError(
                        "OpenAI API error",
                        details=str(e)
                    ) from e
            
            except Exception as e:
                # Unknown error - retry with backoff
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Unexpected error from OpenAI, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self._max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        "Failed to get embedding from OpenAI",
                        details=str(e)
                    ) from e
        
        # This should never be reached, but satisfies type checker
        raise EmbeddingError(
            "Failed to get embedding from OpenAI",
            details=str(last_exception)
        )
    
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
            embeddings = OpenAIEmbeddings()
            vector = embeddings.embed_query("What is federalism?")
            print(len(vector))  # 1536
            ```
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        embeddings = self._create_embedding([text.strip()])
        return embeddings[0]
    
    def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a list of documents into vector representations.
        
        This method batches texts for efficient API usage. OpenAI supports
        batch embedding up to 100 texts per request.
        
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
            embeddings = OpenAIEmbeddings()
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
        
        # Strip all texts
        stripped_texts = [text.strip() for text in texts]
        
        # Process in batches
        all_embeddings: list[EmbeddingVector] = []
        
        for i in range(0, len(stripped_texts), self._max_batch_size):
            batch = stripped_texts[i:i + self._max_batch_size]
            logger.debug(
                f"Embedding batch {i // self._max_batch_size + 1} "
                f"({len(batch)} texts)"
            )
            batch_embeddings = self._create_embedding(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def __repr__(self) -> str:
        """Return string representation of the OpenAIEmbeddings instance."""
        return (
            f"OpenAIEmbeddings("
            f"model={self._model!r}, "
            f"dimension={self.dimension})"
        )
