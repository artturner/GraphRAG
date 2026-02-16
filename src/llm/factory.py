"""Factory for creating LLM provider instances.

This module provides a factory class for creating LLM provider instances
based on configuration. It supports multiple providers including OpenAI,
AWS Bedrock (Claude), and Ollama.

Example:
    ```python
    from src.llm import LLMFactory
    from src.config import LLMConfig
    
    config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
    llm = LLMFactory.get_llm(config)
    ```
"""

import logging
from typing import Literal

from src.config import LLMConfig
from src.exceptions import ConfigurationError
from src.llm.base import BaseLLM
from src.llm.bedrock import BedrockLLM
from src.llm.ollama import OllamaLLM
from src.llm.openai_llm import OpenAILLM

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM provider instances.
    
    This factory class provides a centralized way to create LLM provider
    instances based on configuration. It abstracts away the details of
    instantiating different provider types.
    
    Supported providers:
        - "openai": OpenAI GPT models (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
        - "bedrock" or "bedrock_claude": AWS Bedrock Claude models
        - "ollama": Local Ollama models (llama2, mistral, etc.)
    
    Example:
        ```python
        # Create OpenAI LLM
        config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
        llm = LLMFactory.get_llm(config)
        
        # Create Bedrock Claude LLM
        config = LLMConfig(provider="bedrock_claude", model_name="anthropic.claude-3-sonnet-20240229-v1:0")
        llm = LLMFactory.get_llm(config)
        
        # Create Ollama LLM
        config = LLMConfig(provider="ollama", model_name="llama2")
        llm = LLMFactory.get_llm(config)
        ```
    """
    
    # Mapping of provider names to their implementation classes
    _PROVIDERS: dict[str, type[BaseLLM]] = {
        "openai": OpenAILLM,
        "bedrock": BedrockLLM,
        "bedrock_claude": BedrockLLM,
        "ollama": OllamaLLM,
    }
    
    @classmethod
    def get_llm(cls, config: LLMConfig) -> BaseLLM:
        """Create an LLM provider instance based on configuration.
        
        Args:
            config: LLMConfig instance specifying the provider and model.
            
        Returns:
            An instance of the appropriate BaseLLM subclass.
            
        Raises:
            ConfigurationError: If the provider is not supported or
                configuration is invalid.
        
        Example:
            ```python
            config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
            llm = LLMFactory.get_llm(config)
            print(f"Model: {llm.model_name}")  # gpt-4-turbo
            ```
        """
        provider = config.provider
        
        # Check if provider is supported
        if provider not in cls._PROVIDERS:
            valid_providers = sorted(cls._PROVIDERS.keys())
            raise ConfigurationError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Valid providers are: {valid_providers}"
            )
        
        # Get the provider class
        provider_class = cls._PROVIDERS[provider]
        
        # Create instance based on provider type
        logger.info(f"Creating LLM provider: {provider} with model: {config.model_name}")
        
        if provider == "openai":
            return cls._create_openai_llm(config, provider_class)
        elif provider in ("bedrock", "bedrock_claude"):
            return cls._create_bedrock_llm(config, provider_class)
        elif provider == "ollama":
            return cls._create_ollama_llm(config, provider_class)
        else:
            # This should not happen due to the check above, but just in case
            raise ConfigurationError(f"Unknown provider: {provider}")
    
    @classmethod
    def _create_openai_llm(
        cls,
        config: LLMConfig,
        provider_class: type[OpenAILLM],
    ) -> OpenAILLM:
        """Create an OpenAILLM instance.
        
        Args:
            config: LLMConfig with provider settings.
            provider_class: The OpenAILLM class to instantiate.
            
        Returns:
            A configured OpenAILLM instance.
        """
        return provider_class(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    
    @classmethod
    def _create_bedrock_llm(
        cls,
        config: LLMConfig,
        provider_class: type[BedrockLLM],
    ) -> BedrockLLM:
        """Create a BedrockLLM instance.
        
        Args:
            config: LLMConfig with provider settings.
            provider_class: The BedrockLLM class to instantiate.
            
        Returns:
            A configured BedrockLLM instance.
        """
        return provider_class(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    
    @classmethod
    def _create_ollama_llm(
        cls,
        config: LLMConfig,
        provider_class: type[OllamaLLM],
    ) -> OllamaLLM:
        """Create an OllamaLLM instance.
        
        Args:
            config: LLMConfig with provider settings.
            provider_class: The OllamaLLM class to instantiate.
            
        Returns:
            A configured OllamaLLM instance.
        """
        return provider_class(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider names.
        
        Returns:
            A sorted list of supported provider names.
        
        Example:
            ```python
            providers = LLMFactory.get_supported_providers()
            print(providers)  # ['bedrock', 'bedrock_claude', 'ollama', 'openai']
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
            if LLMFactory.is_provider_supported("openai"):
                config = LLMConfig(provider="openai")
                llm = LLMFactory.get_llm(config)
            ```
        """
        return provider in cls._PROVIDERS
