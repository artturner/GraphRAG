"""Embedding providers for generating vector embeddings from text.

This module provides the base classes and protocols for embedding providers,
as well as concrete implementations for local and remote embedding services.

Classes:
    BaseEmbeddings: Abstract base class for embedding providers.
    EmbeddingCache: Protocol for embedding cache implementations.
    LocalEmbeddings: Local sentence-transformers embedding provider.
    OpenAIEmbeddings: OpenAI text-embedding-3 embedding provider.
    BedrockEmbeddings: AWS Bedrock Titan embedding provider.
    DiskEmbeddingCache: Disk-based persistent embedding cache.
    CachedEmbeddings: Wrapper that adds transparent caching to any provider.
    EmbeddingsFactory: Factory for creating embedding provider instances.

Example:
    ```python
    from src.embeddings import EmbeddingsFactory, CachedEmbeddings, DiskEmbeddingCache
    from src.config import EmbeddingsConfig
    
    # Create embeddings using factory
    config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
    embeddings = EmbeddingsFactory.get_embeddings(config)
    
    # Wrap with cache for transparent caching
    cache = DiskEmbeddingCache("./cache/embeddings")
    cached_embeddings = CachedEmbeddings(embeddings, cache)
    
    # Use cached embeddings - first call computes, subsequent calls use cache
    vector = cached_embeddings.embed_query("Hello world")
    ```
"""

from src.embeddings.base import BaseEmbeddings, EmbeddingCache
from src.embeddings.local import LocalEmbeddings
from src.embeddings.openai_emb import OpenAIEmbeddings
from src.embeddings.bedrock import BedrockEmbeddings
from src.embeddings.cache import DiskEmbeddingCache
from src.embeddings.cached import CachedEmbeddings
from src.embeddings.factory import EmbeddingsFactory

__all__ = [
    # Base classes and protocols
    "BaseEmbeddings",
    "EmbeddingCache",
    # Concrete implementations
    "LocalEmbeddings",
    "OpenAIEmbeddings",
    "BedrockEmbeddings",
    # Cache implementations
    "DiskEmbeddingCache",
    # Wrapper classes
    "CachedEmbeddings",
    # Factory
    "EmbeddingsFactory",
]
