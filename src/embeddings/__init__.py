"""Embedding providers for generating vector embeddings from text.

This module provides the base classes and protocols for embedding providers,
as well as concrete implementations for local and remote embedding services.
"""

from src.embeddings.base import BaseEmbeddings, EmbeddingCache
from src.embeddings.local import LocalEmbeddings
from src.embeddings.openai_emb import OpenAIEmbeddings

__all__ = ["BaseEmbeddings", "EmbeddingCache", "LocalEmbeddings", "OpenAIEmbeddings"]
