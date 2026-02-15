"""Factory for creating embedding provider instances.

This module provides a factory class for creating embedding provider instances
based on configuration. It supports multiple providers including local
sentence-transformers, OpenAI, and AWS Bedrock.

Example:
    ```python
    from src.embeddings import EmbeddingsFactory
    from src.config import EmbeddingsConfig
    
    config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
    embeddings = EmbeddingsFactory.get_embeddings(config)
    ```
"""

import logging
from typing import Literal

from src.config import EmbeddingsConfig
from src.embeddings.base import BaseEmbeddings
from src.embeddings.local import LocalEmbeddings
from src.embeddings.openai_emb import OpenAIEmbeddings
from src.embeddings.bedrock import BedrockEmbeddings
from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class EmbeddingsFactory:
    """Factory for creating embedding provider instances.
    
    This factory class provides a centralized way to create embedding provider
    instances based on configuration. It abstracts away the details of
    instantiating different provider types.
    
    Supported providers:
        - "local" or "local_st": Local sentence-transformers embeddings
        - "openai": OpenAI text-embedding-3 models
        - "bedrock" or "bedrock_titan": AWS Bedrock Titan embeddings
    
    Example:
        ```python
        # Create local embeddings
        config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
        embeddings = EmbeddingsFactory.get_embeddings(config)
        
        # Create OpenAI embeddings
        config = EmbeddingsConfig(provider="openai", model_name="text-embedding-3-small")
        embeddings = EmbeddingsFactory.get_embeddings(config)
        
        # Create Bedrock embeddings
        config = EmbeddingsConfig(provider="bedrock", model_name="amazon.titan-embed-text-v1")
        embeddings = EmbeddingsFactory.get_embeddings(config)
        ```
    """
    
    # Mapping of provider names to their implementation classes
    _PROVIDERS: dict[str, type[BaseEmbeddings]] = {
        "local": LocalEmbeddings,
        "local_st": LocalEmbeddings,
        "openai": OpenAIEmbeddings,
        "bedrock": BedrockEmbeddings,
        "bedrock_titan": BedrockEmbeddings,
    }
    
    @classmethod
    def get_embeddings(cls, config: EmbeddingsConfig) -> BaseEmbeddings:
        """Create an embedding provider instance based on configuration.
        
        Args:
            config: EmbeddingsConfig instance specifying the provider and model.
            
        Returns:
            An instance of the appropriate BaseEmbeddings subclass.
            
        Raises:
            ConfigurationError: If the provider is not supported or
                configuration is invalid.
        
        Example:
            ```python
            config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
            embeddings = EmbeddingsFactory.get_embeddings(config)
            print(f"Dimension: {embeddings.dimension}")  # 384
            ```
        """
        provider = config.provider
        
        # Check if provider is supported
        if provider not in cls._PROVIDERS:
            valid_providers = sorted(cls._PROVIDERS.keys())
            raise ConfigurationError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Valid providers are: {valid_providers}"
            )
        
        # Get the provider class
        provider_class = cls._PROVIDERS[provider]
        
        # Create instance based on provider type
        logger.info(f"Creating embedding provider: {provider} with model: {config.model_name}")
        
        if provider in ("local", "local_st"):
            return cls._create_local_embeddings(config, provider_class)
        elif provider == "openai":
            return cls._create_openai_embeddings(config, provider_class)
        elif provider in ("bedrock", "bedrock_titan"):
            return cls._create_bedrock_embeddings(config, provider_class)
        else:
            # This should not happen due to the check above, but just in case
            raise ConfigurationError(f"Unknown provider: {provider}")
    
    @classmethod
    def _create_local_embeddings(
        cls,
        config: EmbeddingsConfig,
        provider_class: type[LocalEmbeddings],
    ) -> LocalEmbeddings:
        """Create a LocalEmbeddings instance.
        
        Args:
            config: EmbeddingsConfig with provider settings.
            provider_class: The LocalEmbeddings class to instantiate.
            
        Returns:
            A configured LocalEmbeddings instance.
        """
        # Extract model name - handle both full HuggingFace path and short name
        model_name = config.model_name
        if "/" in model_name:
            # Full HuggingFace path like "sentence-transformers/all-MiniLM-L6-v2"
            # LocalEmbeddings expects just the model name part after the slash
            # or the full path - both work with sentence-transformers
            pass
        
        return provider_class(
            model_name=model_name,
        )
    
    @classmethod
    def _create_openai_embeddings(
        cls,
        config: EmbeddingsConfig,
        provider_class: type[OpenAIEmbeddings],
    ) -> OpenAIEmbeddings:
        """Create an OpenAIEmbeddings instance.
        
        Args:
            config: EmbeddingsConfig with provider settings.
            provider_class: The OpenAIEmbeddings class to instantiate.
            
        Returns:
            A configured OpenAIEmbeddings instance.
        """
        return provider_class(
            model=config.model_name,
        )
    
    @classmethod
    def _create_bedrock_embeddings(
        cls,
        config: EmbeddingsConfig,
        provider_class: type[BedrockEmbeddings],
    ) -> BedrockEmbeddings:
        """Create a BedrockEmbeddings instance.
        
        Args:
            config: EmbeddingsConfig with provider settings.
            provider_class: The BedrockEmbeddings class to instantiate.
            
        Returns:
            A configured BedrockEmbeddings instance.
        """
        return provider_class(
            model_id=config.model_name,
        )
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider names.
        
        Returns:
            A sorted list of supported provider names.
        
        Example:
            ```python
            providers = EmbeddingsFactory.get_supported_providers()
            print(providers)  # ['bedrock', 'bedrock_titan', 'local', 'local_st', 'openai']
            ```
        """
        return sorted(cls._PROVIDERS.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """Check if a provider is supported.
        
        Args:
            provider: The provider name to check.
            
        Returns:
            True if the provider is supported, False otherwise.
        
        Example:
            ```python
            if EmbeddingsFactory.is_provider_supported("local"):
                config = EmbeddingsConfig(provider="local")
                embeddings = EmbeddingsFactory.get_embeddings(config)
            ```
        """
        return provider in cls._PROVIDERS