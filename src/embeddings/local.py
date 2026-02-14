"""Local embedding provider using sentence-transformers.

This module provides a local embedding implementation using the sentence-transformers
library, which runs entirely on the local machine without requiring external API calls.

The default model is "all-MiniLM-L6-v2", which offers a good balance between speed
and quality for most use cases.
"""

import logging
from typing import Literal

from src.embeddings.base import BaseEmbeddings
from src.exceptions import EmbeddingError
from src.types import EmbeddingVector

logger = logging.getLogger(__name__)


class LocalEmbeddings(BaseEmbeddings):
    """Local embedding provider using sentence-transformers.
    
    This class provides embedding functionality using the sentence-transformers
    library, which runs locally without requiring external API calls. It supports
    GPU acceleration when available and handles batch embedding efficiently.
    
    The default model "all-MiniLM-L6-v2" produces 384-dimensional vectors and
    offers a good balance between speed and quality.
    
    Attributes:
        model_name: Name of the sentence-transformers model to use.
        dimension: The dimensionality of the embedding vectors.
        device: The device being used (cpu, cuda, or mps).
        normalize_embeddings: Whether to normalize output embeddings.
    
    Example:
        ```python
        # Basic usage with default model
        embeddings = LocalEmbeddings()
        vectors = embeddings.embed_documents(["Hello world", "Test sentence"])
        print(f"Dimension: {embeddings.dimension}")  # 384
        
        # Using a different model
        embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2")
        print(f"Dimension: {embeddings.dimension}")  # 768
        
        # Check if GPU is being used
        print(f"Device: {embeddings.device}")
        ```
    """
    
    # Known model dimensions for common models
    # This allows dimension property to work without loading the model first
    _MODEL_DIMENSIONS: dict[str, int] = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "multi-qa-mpnet-base-dot-v1": 768,
        "distilbert-base-nli-mean-tokens": 768,
        "bert-base-nli-mean-tokens": 768,
        "roberta-base-nli-stsb-mean-tokens": 768,
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        cache_folder: str | None = None,
    ) -> None:
        """Initialize the local embedding provider.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                Defaults to "all-MiniLM-L6-v2" (384 dimensions, fast, good quality).
            device: Device to use for inference. Options are:
                - "auto": Automatically detect and use the best available device.
                - "cpu": Force CPU usage.
                - "cuda": Force CUDA GPU usage.
                - "mps": Force Apple Metal Performance Shaders (M1/M2 Macs).
            normalize_embeddings: Whether to normalize output embeddings to unit length.
                Normalized embeddings are useful for cosine similarity comparisons.
            batch_size: Batch size for embedding multiple documents. Larger batches
                are faster but use more memory.
            cache_folder: Optional folder to cache downloaded models.
                If None, uses the default sentence-transformers cache.
        
        Raises:
            EmbeddingError: If the model fails to load.
        
        Example:
            ```python
            # Default configuration
            embeddings = LocalEmbeddings()
            
            # Force CPU usage
            embeddings = LocalEmbeddings(device="cpu")
            
            # Use larger model with custom batch size
            embeddings = LocalEmbeddings(
                model_name="all-mpnet-base-v2",
                batch_size=64
            )
            ```
        """
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size
        self._cache_folder = cache_folder
        self._model = None
        self._dimension: int | None = None
        self._device: str | None = None
        self._requested_device = device
        
        # Set dimension from known models if available
        if model_name in self._MODEL_DIMENSIONS:
            self._dimension = self._MODEL_DIMENSIONS[model_name]
    
    def _get_device(self) -> str:
        """Determine the best available device for inference.
        
        Returns:
            The device string: "cuda", "mps", or "cpu".
        """
        try:
            import torch
            
            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                return "cuda"
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
        
        return "cpu"
    
    def _load_model(self) -> None:
        """Load the sentence-transformers model.
        
        This method lazily loads the model on first use to avoid
        unnecessary memory consumption.
        
        Raises:
            EmbeddingError: If the model fails to load.
        """
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Determine device
            if self._requested_device == "auto":
                self._device = self._get_device()
            else:
                self._device = self._requested_device
            
            logger.info(
                f"Loading sentence-transformers model: {self._model_name} "
                f"on device: {self._device}"
            )
            
            # Load the model
            model_kwargs = {}
            if self._cache_folder:
                model_kwargs["cache_folder"] = self._cache_folder
            
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                **model_kwargs
            )
            
            # Get actual dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Model loaded successfully. Dimension: {self._dimension}"
            )
            
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers library not installed",
                details="Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load model: {self._model_name}",
                details=str(e)
            ) from e
    
    @property
    def model_name(self) -> str:
        """Return the name of the sentence-transformers model.
        
        Returns:
            The model name string.
        """
        return self._model_name
    
    @property
    def device(self) -> str:
        """Return the device being used for inference.
        
        Returns:
            The device string: "cuda", "mps", or "cpu".
        """
        if self._device is None:
            self._load_model()
        return self._device  # type: ignore
    
    @property
    def normalize_embeddings(self) -> bool:
        """Return whether embeddings are normalized.
        
        Returns:
            True if embeddings are normalized to unit length.
        """
        return self._normalize_embeddings
    
    @property
    def batch_size(self) -> int:
        """Return the batch size for embedding documents.
        
        Returns:
            The batch size integer.
        """
        return self._batch_size
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.
        
        Returns:
            The number of dimensions in each embedding vector.
        
        Raises:
            EmbeddingError: If the dimension cannot be determined.
        """
        if self._dimension is None:
            self._load_model()
        return self._dimension  # type: ignore
    
    def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a list of documents into vector representations.
        
        This method efficiently processes multiple documents in batches,
        utilizing GPU acceleration when available.
        
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
            embeddings = LocalEmbeddings()
            texts = ["Hello world", "Goodbye world"]
            vectors = embeddings.embed_documents(texts)
            print(len(vectors))  # 2
            print(len(vectors[0]))  # 384 (dimension)
            ```
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("all texts must be strings")
        
        if not all(t.strip() for t in texts):
            raise ValueError("texts cannot contain empty strings")
        
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Encode all texts in batches
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            raise EmbeddingError(
                "Failed to generate document embeddings",
                details=str(e)
            ) from e
    
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single query into a vector representation.
        
        This method is optimized for embedding a single query text,
        typically used during retrieval operations.
        
        Args:
            text: The query text to embed.
            
        Returns:
            An embedding vector representing the query.
        
        Raises:
            EmbeddingError: If the embedding operation fails.
            ValueError: If text is empty or invalid.
        
        Example:
            ```python
            embeddings = LocalEmbeddings()
            query = "What is machine learning?"
            vector = embeddings.embed_query(query)
            print(len(vector))  # 384 (dimension)
            ```
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace only")
        
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Encode single text
            embedding = self._model.encode(
                text,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            return embedding.tolist()
            
        except Exception as e:
            raise EmbeddingError(
                "Failed to generate query embedding",
                details=str(e)
            ) from e
    
    def __repr__(self) -> str:
        """Return a string representation of the LocalEmbeddings instance.
        
        Returns:
            A string representation including model name and device.
        """
        device_str = self._device if self._device else "not loaded"
        return (
            f"LocalEmbeddings("
            f"model_name={self._model_name!r}, "
            f"device={device_str!r}, "
            f"dimension={self._dimension})"
        )