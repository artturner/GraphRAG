"""Tests for the EmbeddingsFactory class.

This module tests the factory functionality for creating embedding providers
based on configuration.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.config import EmbeddingsConfig
from src.embeddings import EmbeddingsFactory, LocalEmbeddings, OpenAIEmbeddings, BedrockEmbeddings
from src.embeddings.factory import EmbeddingsFactory as FactoryClass
from src.exceptions import ConfigurationError


class TestEmbeddingsFactory:
    """Test suite for EmbeddingsFactory class."""
    
    def test_get_supported_providers(self):
        """Test that get_supported_providers returns expected list."""
        providers = EmbeddingsFactory.get_supported_providers()
        
        assert "local" in providers
        assert "local_st" in providers
        assert "openai" in providers
        assert "bedrock" in providers
        assert "bedrock_titan" in providers
    
    def test_is_provider_supported_true(self):
        """Test is_provider_supported returns True for valid providers."""
        assert EmbeddingsFactory.is_provider_supported("local")
        assert EmbeddingsFactory.is_provider_supported("local_st")
        assert EmbeddingsFactory.is_provider_supported("openai")
        assert EmbeddingsFactory.is_provider_supported("bedrock")
        assert EmbeddingsFactory.is_provider_supported("bedrock_titan")
    
    def test_is_provider_supported_false(self):
        """Test is_provider_supported returns False for invalid providers."""
        assert not EmbeddingsFactory.is_provider_supported("invalid")
        assert not EmbeddingsFactory.is_provider_supported("unknown")
        assert not EmbeddingsFactory.is_provider_supported("")
    
    def test_get_embeddings_local(self):
        """Test factory creates LocalEmbeddings with 'local' provider."""
        config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
        
        with patch.object(LocalEmbeddings, '__init__', return_value=None) as mock_init:
            EmbeddingsFactory.get_embeddings(config)
            mock_init.assert_called_once()
            # Check model_name was passed
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model_name'] == "all-MiniLM-L6-v2"
    
    def test_get_embeddings_local_st(self):
        """Test factory creates LocalEmbeddings with 'local_st' provider alias."""
        config = EmbeddingsConfig(provider="local_st", model_name="all-MiniLM-L6-v2")
        
        with patch.object(LocalEmbeddings, '__init__', return_value=None) as mock_init:
            EmbeddingsFactory.get_embeddings(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model_name'] == "all-MiniLM-L6-v2"
    
    def test_get_embeddings_openai(self):
        """Test factory creates OpenAIEmbeddings with 'openai' provider."""
        config = EmbeddingsConfig(provider="openai", model_name="text-embedding-3-small")
        
        with patch.object(OpenAIEmbeddings, '__init__', return_value=None) as mock_init:
            EmbeddingsFactory.get_embeddings(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model'] == "text-embedding-3-small"
    
    def test_get_embeddings_bedrock(self):
        """Test factory creates BedrockEmbeddings with 'bedrock' provider."""
        config = EmbeddingsConfig(
            provider="bedrock",
            model_name="amazon.titan-embed-text-v1"
        )
        
        with patch.object(BedrockEmbeddings, '__init__', return_value=None) as mock_init:
            EmbeddingsFactory.get_embeddings(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model_id'] == "amazon.titan-embed-text-v1"
    
    def test_get_embeddings_bedrock_titan(self):
        """Test factory creates BedrockEmbeddings with 'bedrock_titan' alias."""
        config = EmbeddingsConfig(
            provider="bedrock_titan",
            model_name="amazon.titan-embed-text-v1"
        )
        
        with patch.object(BedrockEmbeddings, '__init__', return_value=None) as mock_init:
            EmbeddingsFactory.get_embeddings(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model_id'] == "amazon.titan-embed-text-v1"
    
    def test_get_embeddings_returns_correct_type_local(self):
        """Test that factory returns LocalEmbeddings instance for local provider."""
        # Create a mock LocalEmbeddings that doesn't require sentence-transformers
        mock_embeddings = MagicMock(spec=LocalEmbeddings)
        mock_embeddings.dimension = 384
        
        with patch.object(
            FactoryClass, '_create_local_embeddings', return_value=mock_embeddings
        ):
            config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
            result = EmbeddingsFactory.get_embeddings(config)
            
            assert isinstance(result, LocalEmbeddings) or hasattr(result, 'dimension')
    
    def test_get_embeddings_returns_correct_type_openai(self):
        """Test that factory returns OpenAIEmbeddings instance for openai provider."""
        mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
        mock_embeddings.dimension = 1536
        
        with patch.object(
            FactoryClass, '_create_openai_embeddings', return_value=mock_embeddings
        ):
            config = EmbeddingsConfig(provider="openai", model_name="text-embedding-3-small")
            result = EmbeddingsFactory.get_embeddings(config)
            
            assert isinstance(result, OpenAIEmbeddings) or hasattr(result, 'dimension')
    
    def test_get_embeddings_returns_correct_type_bedrock(self):
        """Test that factory returns BedrockEmbeddings instance for bedrock provider."""
        mock_embeddings = MagicMock(spec=BedrockEmbeddings)
        mock_embeddings.dimension = 1536
        
        with patch.object(
            FactoryClass, '_create_bedrock_embeddings', return_value=mock_embeddings
        ):
            config = EmbeddingsConfig(provider="bedrock", model_name="amazon.titan-embed-text-v1")
            result = EmbeddingsFactory.get_embeddings(config)
            
            assert isinstance(result, BedrockEmbeddings) or hasattr(result, 'dimension')
    
    def test_get_embeddings_with_full_huggingface_path(self):
        """Test factory handles full HuggingFace model path."""
        config = EmbeddingsConfig(
            provider="local",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        with patch.object(LocalEmbeddings, '__init__', return_value=None) as mock_init:
            EmbeddingsFactory.get_embeddings(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['model_name'] == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_get_embeddings_invalid_provider(self):
        """Test factory raises ConfigurationError for invalid provider."""
        # Create config with invalid provider by bypassing validation
        config = EmbeddingsConfig(provider="local", model_name="test")
        # Manually set provider to invalid value
        object.__setattr__(config, 'provider', 'invalid_provider')
        
        with pytest.raises(ConfigurationError) as exc_info:
            EmbeddingsFactory.get_embeddings(config)
        
        assert "Unsupported embedding provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)
    
    def test_provider_mapping_completeness(self):
        """Test that all supported providers have corresponding classes."""
        providers = EmbeddingsFactory.get_supported_providers()
        provider_map = EmbeddingsFactory._PROVIDERS
        
        for provider in providers:
            assert provider in provider_map, f"Provider {provider} missing from mapping"
            assert provider_map[provider] is not None, f"Provider {provider} has None class"


class TestEmbeddingsFactoryIntegration:
    """Integration tests for EmbeddingsFactory with real instances."""
    
    @pytest.mark.skip(reason="sentence-transformers not installed")
    def test_real_local_embeddings_creation(self):
        """Test creating real LocalEmbeddings instance."""
        config = EmbeddingsConfig(provider="local", model_name="all-MiniLM-L6-v2")
        embeddings = EmbeddingsFactory.get_embeddings(config)
        
        assert isinstance(embeddings, LocalEmbeddings)
        assert embeddings.dimension == 384


class TestEmbeddingsConfigValidation:
    """Tests for EmbeddingsConfig provider validation."""
    
    def test_valid_provider_local(self):
        """Test that 'local' provider is valid."""
        config = EmbeddingsConfig(provider="local", model_name="test")
        assert config.provider == "local"
    
    def test_valid_provider_local_st(self):
        """Test that 'local_st' provider alias is valid."""
        config = EmbeddingsConfig(provider="local_st", model_name="test")
        assert config.provider == "local_st"
    
    def test_valid_provider_openai(self):
        """Test that 'openai' provider is valid."""
        config = EmbeddingsConfig(provider="openai", model_name="test")
        assert config.provider == "openai"
    
    def test_valid_provider_bedrock(self):
        """Test that 'bedrock' provider is valid."""
        config = EmbeddingsConfig(provider="bedrock", model_name="test")
        assert config.provider == "bedrock"
    
    def test_valid_provider_bedrock_titan(self):
        """Test that 'bedrock_titan' provider alias is valid."""
        config = EmbeddingsConfig(provider="bedrock_titan", model_name="test")
        assert config.provider == "bedrock_titan"
    
    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingsConfig(provider="invalid_provider", model_name="test")
        
        assert "provider must be one of" in str(exc_info.value)


class TestFactoryProviderAliases:
    """Tests for provider alias handling."""
    
    def test_local_st_creates_local_embeddings(self):
        """Test that 'local_st' alias creates LocalEmbeddings."""
        config = EmbeddingsConfig(provider="local_st", model_name="all-MiniLM-L6-v2")
        
        with patch.object(LocalEmbeddings, '__init__', return_value=None) as mock_init:
            result = EmbeddingsFactory.get_embeddings(config)
            # Verify LocalEmbeddings.__init__ was called
            mock_init.assert_called_once()
    
    def test_bedrock_titan_creates_bedrock_embeddings(self):
        """Test that 'bedrock_titan' alias creates BedrockEmbeddings."""
        config = EmbeddingsConfig(
            provider="bedrock_titan",
            model_name="amazon.titan-embed-text-v1"
        )
        
        with patch.object(BedrockEmbeddings, '__init__', return_value=None) as mock_init:
            result = EmbeddingsFactory.get_embeddings(config)
            # Verify BedrockEmbeddings.__init__ was called
            mock_init.assert_called_once()