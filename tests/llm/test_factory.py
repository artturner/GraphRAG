"""Tests for the LLMFactory class.

This module tests the factory functionality for creating LLM providers
based on configuration.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.config import LLMConfig
from src.llm import LLMFactory, OpenAILLM, BedrockLLM, OllamaLLM, BaseLLM
from src.llm.factory import LLMFactory as FactoryClass
from src.exceptions import ConfigurationError


class TestLLMFactory:
    """Test suite for LLMFactory class."""
    
    def test_get_supported_providers(self):
        """Test that get_supported_providers returns expected list."""
        providers = LLMFactory.get_supported_providers()
        
        assert "openai" in providers
        assert "bedrock" in providers
        assert "bedrock_claude" in providers
        assert "ollama" in providers
    
    def test_is_provider_supported_true(self):
        """Test is_provider_supported returns True for valid providers."""
        assert LLMFactory.is_provider_supported("openai")
        assert LLMFactory.is_provider_supported("bedrock")
        assert LLMFactory.is_provider_supported("bedrock_claude")
        assert LLMFactory.is_provider_supported("ollama")
    
    def test_is_provider_supported_false(self):
        """Test is_provider_supported returns False for invalid providers."""
        assert not LLMFactory.is_provider_supported("invalid")
        assert not LLMFactory.is_provider_supported("unknown")
        assert not LLMFactory.is_provider_supported("")
    
    def test_get_llm_openai(self):
        """Test factory creates OpenAILLM with 'openai' provider."""
        config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
        
        with patch.object(OpenAILLM, '__init__', return_value=None) as mock_init:
            LLMFactory.get_llm(config)
            mock_init.assert_called_once()
            # Check parameters were passed
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model'] == "gpt-4-turbo"
            assert call_kwargs['temperature'] == config.temperature
            assert call_kwargs['max_tokens'] == config.max_tokens
    
    def test_get_llm_bedrock(self):
        """Test factory creates BedrockLLM with 'bedrock' provider."""
        config = LLMConfig(
            provider="bedrock",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        
        with patch.object(BedrockLLM, '__init__', return_value=None) as mock_init:
            LLMFactory.get_llm(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model'] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert call_kwargs['temperature'] == config.temperature
            assert call_kwargs['max_tokens'] == config.max_tokens
    
    def test_get_llm_bedrock_claude(self):
        """Test factory creates BedrockLLM with 'bedrock_claude' alias."""
        config = LLMConfig(
            provider="bedrock_claude",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        
        with patch.object(BedrockLLM, '__init__', return_value=None) as mock_init:
            LLMFactory.get_llm(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model'] == "anthropic.claude-3-sonnet-20240229-v1:0"
    
    def test_get_llm_ollama(self):
        """Test factory creates OllamaLLM with 'ollama' provider."""
        config = LLMConfig(provider="ollama", model_name="llama2")
        
        with patch.object(OllamaLLM, '__init__', return_value=None) as mock_init:
            LLMFactory.get_llm(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model'] == "llama2"
            assert call_kwargs['temperature'] == config.temperature
            assert call_kwargs['max_tokens'] == config.max_tokens
    
    def test_get_llm_returns_correct_type_openai(self):
        """Test factory returns OpenAILLM instance for 'openai' provider."""
        config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
        
        with patch.object(OpenAILLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, OpenAILLM)
    
    def test_get_llm_returns_correct_type_bedrock(self):
        """Test factory returns BedrockLLM instance for 'bedrock' provider."""
        config = LLMConfig(
            provider="bedrock",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        
        with patch.object(BedrockLLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, BedrockLLM)
    
    def test_get_llm_returns_correct_type_bedrock_claude(self):
        """Test factory returns BedrockLLM instance for 'bedrock_claude' provider."""
        config = LLMConfig(
            provider="bedrock_claude",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        
        with patch.object(BedrockLLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, BedrockLLM)
    
    def test_get_llm_returns_correct_type_ollama(self):
        """Test factory returns OllamaLLM instance for 'ollama' provider."""
        config = LLMConfig(provider="ollama", model_name="llama2")
        
        with patch.object(OllamaLLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, OllamaLLM)
    
    def test_get_llm_returns_base_llm_type(self):
        """Test factory returns BaseLLM type for all providers."""
        # Test OpenAI
        config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
        with patch.object(OpenAILLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, BaseLLM)
        
        # Test Bedrock
        config = LLMConfig(provider="bedrock", model_name="anthropic.claude-3-sonnet-20240229-v1:0")
        with patch.object(BedrockLLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, BaseLLM)
        
        # Test Ollama
        config = LLMConfig(provider="ollama", model_name="llama2")
        with patch.object(OllamaLLM, '__init__', return_value=None):
            llm = LLMFactory.get_llm(config)
            assert isinstance(llm, BaseLLM)
    
    def test_get_llm_with_custom_temperature(self):
        """Test factory passes custom temperature to LLM."""
        config = LLMConfig(provider="openai", model_name="gpt-4-turbo", temperature=0.7)
        
        with patch.object(OpenAILLM, '__init__', return_value=None) as mock_init:
            LLMFactory.get_llm(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['temperature'] == 0.7
    
    def test_get_llm_with_custom_max_tokens(self):
        """Test factory passes custom max_tokens to LLM."""
        config = LLMConfig(provider="openai", model_name="gpt-4-turbo", max_tokens=2048)
        
        with patch.object(OpenAILLM, '__init__', return_value=None) as mock_init:
            LLMFactory.get_llm(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['max_tokens'] == 2048
    
    def test_get_llm_unsupported_provider_raises_error(self):
        """Test factory raises ConfigurationError for unsupported provider."""
        # Create config with invalid provider by bypassing validation
        config = MagicMock(spec=LLMConfig)
        config.provider = "unsupported_provider"
        config.model_name = "some-model"
        config.temperature = 0.0
        config.max_tokens = 1024
        
        with pytest.raises(ConfigurationError) as exc_info:
            LLMFactory.get_llm(config)
        
        assert "Unsupported LLM provider" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)
    
    def test_factory_class_alias(self):
        """Test that FactoryClass alias works correctly."""
        # This test ensures the import alias works
        assert FactoryClass is LLMFactory
    
    def test_providers_mapping_is_complete(self):
        """Test that all supported providers have a corresponding class."""
        providers = LLMFactory.get_supported_providers()
        
        for provider in providers:
            assert provider in LLMFactory._PROVIDERS
            assert LLMFactory._PROVIDERS[provider] in [OpenAILLM, BedrockLLM, OllamaLLM]
    
    def test_get_llm_openai_with_different_models(self):
        """Test factory creates OpenAILLM with different model names."""
        models = ["gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
        
        for model_name in models:
            config = LLMConfig(provider="openai", model_name=model_name)
            
            with patch.object(OpenAILLM, '__init__', return_value=None) as mock_init:
                LLMFactory.get_llm(config)
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs['model'] == model_name
    
    def test_get_llm_ollama_with_different_models(self):
        """Test factory creates OllamaLLM with different model names."""
        models = ["llama2", "mistral", "codellama", "llama3"]
        
        for model_name in models:
            config = LLMConfig(provider="ollama", model_name=model_name)
            
            with patch.object(OllamaLLM, '__init__', return_value=None) as mock_init:
                LLMFactory.get_llm(config)
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs['model'] == model_name
    
    def test_get_llm_bedrock_with_different_models(self):
        """Test factory creates BedrockLLM with different model names."""
        models = [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
        ]
        
        for model_name in models:
            config = LLMConfig(provider="bedrock", model_name=model_name)
            
            with patch.object(BedrockLLM, '__init__', return_value=None) as mock_init:
                LLMFactory.get_llm(config)
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs['model'] == model_name
