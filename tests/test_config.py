"""Tests for the configuration system."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import (
    CorpusConfig,
    EmbeddingsConfig,
    GraphConfig,
    LLMConfig,
    Settings,
    VectorStoreConfig,
    settings,
)


class TestCorpusConfig:
    """Tests for CorpusConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = CorpusConfig()
        assert config.name == "default"
        assert config.path == "./data"
        assert config.connector_type == "local"

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = CorpusConfig(
            name="test_corpus",
            path="/custom/path",
            connector_type="s3"
        )
        assert config.name == "test_corpus"
        assert config.path == "/custom/path"
        assert config.connector_type == "s3"

    def test_invalid_connector_type(self):
        """Test validation error for invalid connector type."""
        with pytest.raises(ValidationError) as exc_info:
            CorpusConfig(connector_type="invalid")
        assert "connector_type must be one of" in str(exc_info.value)

    def test_valid_connector_types(self):
        """Test all valid connector types."""
        for connector_type in ["local", "s3", "web"]:
            config = CorpusConfig(connector_type=connector_type)
            assert config.connector_type == connector_type


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = VectorStoreConfig()
        assert config.type == "faiss"
        assert config.persist_directory == "./.vectorstore"
        assert config.collection_name == "default"

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = VectorStoreConfig(
            type="chroma",
            persist_directory="/custom/vectors",
            collection_name="test_collection"
        )
        assert config.type == "chroma"
        assert config.persist_directory == "/custom/vectors"
        assert config.collection_name == "test_collection"

    def test_invalid_type(self):
        """Test validation error for invalid vector store type."""
        with pytest.raises(ValidationError) as exc_info:
            VectorStoreConfig(type="invalid")
        assert "type must be one of" in str(exc_info.value)

    def test_valid_types(self):
        """Test all valid vector store types."""
        for store_type in ["faiss", "chroma"]:
            config = VectorStoreConfig(type=store_type)
            assert config.type == store_type


class TestEmbeddingsConfig:
    """Tests for EmbeddingsConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = EmbeddingsConfig()
        assert config.provider == "local"
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimension == 384

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = EmbeddingsConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            dimension=1536
        )
        assert config.provider == "openai"
        assert config.model_name == "text-embedding-3-small"
        assert config.dimension == 1536

    def test_invalid_provider(self):
        """Test validation error for invalid embedding provider."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingsConfig(provider="invalid")
        assert "provider must be one of" in str(exc_info.value)

    def test_invalid_dimension(self):
        """Test validation error for invalid dimension."""
        with pytest.raises(ValidationError):
            EmbeddingsConfig(dimension=0)
        
        with pytest.raises(ValidationError):
            EmbeddingsConfig(dimension=-1)

    def test_valid_providers(self):
        """Test all valid embedding providers."""
        for provider in ["openai", "bedrock", "local"]:
            config = EmbeddingsConfig(provider=provider)
            assert config.provider == provider


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = LLMConfig(
            provider="bedrock",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0.5,
            max_tokens=2048
        )
        assert config.provider == "bedrock"
        assert config.model_name == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_invalid_provider(self):
        """Test validation error for invalid LLM provider."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(provider="invalid")
        assert "provider must be one of" in str(exc_info.value)

    def test_invalid_temperature(self):
        """Test validation error for invalid temperature."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_invalid_max_tokens(self):
        """Test validation error for invalid max_tokens."""
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)
        
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=-1)

    def test_valid_providers(self):
        """Test all valid LLM providers."""
        for provider in ["openai", "bedrock", "ollama"]:
            config = LLMConfig(provider=provider)
            assert config.provider == provider

    def test_temperature_boundaries(self):
        """Test temperature at boundary values."""
        config_min = LLMConfig(temperature=0.0)
        assert config_min.temperature == 0.0
        
        config_max = LLMConfig(temperature=2.0)
        assert config_max.temperature == 2.0


class TestGraphConfig:
    """Tests for GraphConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = GraphConfig()
        assert config.type == "rag"
        assert config.max_retries == 2
        assert config.refusal_threshold == 0.8

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = GraphConfig(
            type="multi_turn",
            max_retries=5,
            refusal_threshold=0.5
        )
        assert config.type == "multi_turn"
        assert config.max_retries == 5
        assert config.refusal_threshold == 0.5

    def test_invalid_type(self):
        """Test validation error for invalid graph type."""
        with pytest.raises(ValidationError) as exc_info:
            GraphConfig(type="invalid")
        assert "type must be one of" in str(exc_info.value)

    def test_invalid_max_retries(self):
        """Test validation error for invalid max_retries."""
        with pytest.raises(ValidationError):
            GraphConfig(max_retries=-1)

    def test_invalid_refusal_threshold(self):
        """Test validation error for invalid refusal_threshold."""
        with pytest.raises(ValidationError):
            GraphConfig(refusal_threshold=-0.1)
        
        with pytest.raises(ValidationError):
            GraphConfig(refusal_threshold=1.1)

    def test_valid_types(self):
        """Test all valid graph types."""
        for graph_type in ["rag", "multi_turn"]:
            config = GraphConfig(type=graph_type)
            assert config.type == graph_type

    def test_refusal_threshold_boundaries(self):
        """Test refusal_threshold at boundary values."""
        config_min = GraphConfig(refusal_threshold=0.0)
        assert config_min.refusal_threshold == 0.0
        
        config_max = GraphConfig(refusal_threshold=1.0)
        assert config_max.refusal_threshold == 1.0


class TestSettings:
    """Tests for the main Settings class."""

    def test_default_settings(self):
        """Test default settings are loaded correctly."""
        test_settings = Settings(config_path="nonexistent.yaml")
        assert test_settings.corpus.name == "default"
        assert test_settings.vectorstore.type == "faiss"
        assert test_settings.embeddings.provider == "local"
        assert test_settings.llm.provider == "openai"
        assert test_settings.graph.type == "rag"

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        test_settings = Settings(config_path="configs/default.yaml")
        assert test_settings.corpus.name == "default"
        assert test_settings.corpus.path == "./data"
        assert test_settings.corpus.connector_type == "local"
        assert test_settings.vectorstore.type == "faiss"
        assert test_settings.embeddings.provider == "local"
        assert test_settings.llm.model_name == "gpt-4"

    def test_yaml_file_not_found(self):
        """Test that missing YAML file uses defaults."""
        test_settings = Settings(config_path="nonexistent_config.yaml")
        # Should use default values
        assert test_settings.corpus.name == "default"
        assert test_settings.vectorstore.type == "faiss"

    def test_singleton_instance_exists(self):
        """Test that the global settings instance exists."""
        from src.config import settings as global_settings
        assert global_settings is not None
        assert hasattr(global_settings, "corpus")
        assert hasattr(global_settings, "llm")

    def test_deep_merge(self):
        """Test the deep merge functionality."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}}
        result = Settings._deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 10, "d": 3}}

    def test_deep_merge_new_keys(self):
        """Test deep merge with new keys in override."""
        base = {"a": 1}
        override = {"b": 2, "c": {"d": 3}}
        result = Settings._deep_merge(base, override)
        assert result == {"a": 1, "b": 2, "c": {"d": 3}}


class TestEnvironmentVariableOverrides:
    """Tests for environment variable overrides."""

    def test_corpus_env_override(self):
        """Test CORPUS_* environment variable overrides."""
        with patch.dict(os.environ, {
            "CORPUS_NAME": "env_corpus",
            "CORPUS_PATH": "/env/path",
            "CORPUS_CONNECTOR_TYPE": "s3"
        }):
            test_settings = Settings(config_path="nonexistent.yaml")
            assert test_settings.corpus.name == "env_corpus"
            assert test_settings.corpus.path == "/env/path"
            assert test_settings.corpus.connector_type == "s3"

    def test_vectorstore_env_override(self):
        """Test VECTORSTORE_* environment variable overrides."""
        with patch.dict(os.environ, {
            "VECTORSTORE_TYPE": "chroma",
            "VECTORSTORE_PERSIST_DIRECTORY": "/env/vectors",
            "VECTORSTORE_COLLECTION_NAME": "env_collection"
        }):
            test_settings = Settings(config_path="nonexistent.yaml")
            assert test_settings.vectorstore.type == "chroma"
            assert test_settings.vectorstore.persist_directory == "/env/vectors"
            assert test_settings.vectorstore.collection_name == "env_collection"

    def test_embeddings_env_override(self):
        """Test EMBEDDINGS_* environment variable overrides."""
        with patch.dict(os.environ, {
            "EMBEDDINGS_PROVIDER": "openai",
            "EMBEDDINGS_MODEL_NAME": "text-embedding-3-small",
            "EMBEDDINGS_DIMENSION": "1536"
        }):
            test_settings = Settings(config_path="nonexistent.yaml")
            assert test_settings.embeddings.provider == "openai"
            assert test_settings.embeddings.model_name == "text-embedding-3-small"
            assert test_settings.embeddings.dimension == 1536

    def test_llm_env_override(self):
        """Test LLM_* environment variable overrides."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "bedrock",
            "LLM_MODEL_NAME": "claude-3",
            "LLM_TEMPERATURE": "0.7",
            "LLM_MAX_TOKENS": "2048"
        }):
            test_settings = Settings(config_path="nonexistent.yaml")
            assert test_settings.llm.provider == "bedrock"
            assert test_settings.llm.model_name == "claude-3"
            assert test_settings.llm.temperature == 0.7
            assert test_settings.llm.max_tokens == 2048

    def test_graph_env_override(self):
        """Test GRAPH_* environment variable overrides."""
        with patch.dict(os.environ, {
            "GRAPH_TYPE": "multi_turn",
            "GRAPH_MAX_RETRIES": "5",
            "GRAPH_REFUSAL_THRESHOLD": "0.9"
        }):
            test_settings = Settings(config_path="nonexistent.yaml")
            assert test_settings.graph.type == "multi_turn"
            assert test_settings.graph.max_retries == 5
            assert test_settings.graph.refusal_threshold == 0.9

    def test_global_env_override(self):
        """Test global environment variable overrides."""
        with patch.dict(os.environ, {
            "DEBUG": "true"
        }):
            test_settings = Settings(config_path="nonexistent.yaml")
            assert test_settings.debug is True


class TestValidationErrors:
    """Tests for validation errors with invalid configuration."""

    def test_invalid_connector_type_env(self):
        """Test validation error with invalid connector type from env."""
        with patch.dict(os.environ, {"CORPUS_CONNECTOR_TYPE": "invalid_type"}):
            with pytest.raises(ValidationError):
                Settings(config_path="nonexistent.yaml")

    def test_invalid_vectorstore_type_env(self):
        """Test validation error with invalid vectorstore type from env."""
        with patch.dict(os.environ, {"VECTORSTORE_TYPE": "invalid_type"}):
            with pytest.raises(ValidationError):
                Settings(config_path="nonexistent.yaml")

    def test_invalid_embeddings_provider_env(self):
        """Test validation error with invalid embeddings provider from env."""
        with patch.dict(os.environ, {"EMBEDDINGS_PROVIDER": "invalid_provider"}):
            with pytest.raises(ValidationError):
                Settings(config_path="nonexistent.yaml")

    def test_invalid_llm_provider_env(self):
        """Test validation error with invalid LLM provider from env."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}):
            with pytest.raises(ValidationError):
                Settings(config_path="nonexistent.yaml")

    def test_invalid_temperature_env(self):
        """Test validation error with invalid temperature from env."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "3.0"}):
            with pytest.raises(ValidationError):
                Settings(config_path="nonexistent.yaml")

    def test_invalid_graph_type_env(self):
        """Test validation error with invalid graph type from env."""
        with patch.dict(os.environ, {"GRAPH_TYPE": "invalid_type"}):
            with pytest.raises(ValidationError):
                Settings(config_path="nonexistent.yaml")


class TestConfigIntegration:
    """Integration tests for the configuration system."""

    def test_yaml_then_env_override(self):
        """Test that env vars override YAML config."""
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "gpt-3.5-turbo"}):
            test_settings = Settings(config_path="configs/default.yaml")
            # YAML has gpt-4, but env should override
            assert test_settings.llm.model_name == "gpt-3.5-turbo"
            # Other values should come from YAML
            assert test_settings.llm.temperature == 0.0

    def test_config_path_attribute(self):
        """Test that config_path is stored correctly."""
        test_settings = Settings(config_path="custom/path.yaml")
        assert test_settings.config_path == "custom/path.yaml"

    def test_reload_functionality(self):
        """Test that reload method exists and can be called."""
        test_settings = Settings(config_path="configs/default.yaml")
        # Should not raise an error
        test_settings.reload()
        assert test_settings.corpus.name == "default"
