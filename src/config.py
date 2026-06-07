"""Configuration system for Grounded GraphRAG Tutor.

This module provides a centralized configuration system using pydantic-settings.
Configuration can be loaded from YAML files and overridden with environment variables.

Usage:
    from src.config import settings
    print(settings.llm.model_name)
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CorpusConfig(BaseSettings):
    """Configuration for document corpus."""
    
    model_config = SettingsConfigDict(env_prefix="CORPUS_")
    
    name: str = Field(default="default", description="Name of the corpus")
    path: str = Field(default="./data", description="Path to document source")
    connector_type: str = Field(default="local", description="Connector type: local, s3, web")
    
    @field_validator("connector_type")
    @classmethod
    def validate_connector_type(cls, v: str) -> str:
        """Validate connector type is one of the supported types."""
        valid_types = {"local", "s3", "web", "json_pages"}
        if v not in valid_types:
            raise ValueError(f"connector_type must be one of {valid_types}, got {v}")
        return v


class IngestionConfig(BaseSettings):
    """Configuration for document ingestion and chunking."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size: int = Field(default=2000, description="Target characters per chunk (~500 tokens)", ge=1)
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks", ge=0)
    chunker: str = Field(default="sentence", description="Chunking strategy: fixed | sentence")
    clean_whitespace: bool = Field(default=True, description="Normalise excessive whitespace")
    clean_html: bool = Field(default=True, description="Strip HTML tags from content")

    @field_validator("chunker")
    @classmethod
    def validate_chunker(cls, v: str) -> str:
        valid = {"fixed", "sentence"}
        if v not in valid:
            raise ValueError(f"chunker must be one of {valid}, got {v}")
        return v


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store."""
    
    model_config = SettingsConfigDict(env_prefix="VECTORSTORE_")
    
    type: str = Field(default="faiss", description="Vector store type: faiss, chroma")
    persist_directory: str = Field(default="./.vectorstore", description="Directory for persistence")
    collection_name: str = Field(default="default", description="Collection name for ChromaDB")
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate vector store type is one of the supported types."""
        valid_types = {"faiss", "chroma"}
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {v}")
        return v


class EmbeddingsConfig(BaseSettings):
    """Configuration for embedding provider."""
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")
    
    provider: str = Field(default="local", description="Provider: openai, bedrock, local")
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for embeddings"
    )
    dimension: int = Field(default=384, description="Embedding dimension", ge=1)
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate embedding provider is one of the supported types."""
        valid_providers = {"openai", "bedrock", "bedrock_titan", "local", "local_st"}
        if v not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got {v}")
        return v


class LLMConfig(BaseSettings):
    """Configuration for LLM provider."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    provider: str = Field(default="openai", description="Provider: openai, bedrock, ollama")
    model_name: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.0, description="Temperature for generation", ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, description="Maximum tokens in response", ge=1)
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider is one of the supported types."""
        valid_providers = {"openai", "bedrock", "bedrock_claude", "ollama"}
        if v not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got {v}")
        return v


class GraphConfig(BaseSettings):
    """Configuration for LangGraph workflow."""
    
    model_config = SettingsConfigDict(env_prefix="GRAPH_")
    
    type: str = Field(default="rag", description="Graph type: rag, multi_turn")
    max_retries: int = Field(default=2, description="Maximum retry attempts", ge=0)
    refusal_threshold: float = Field(
        default=0.8,
        description="Confidence threshold below which to refuse answering",
        ge=0.0,
        le=1.0
    )
    generative_confidence_floor: float = Field(
        default=0.4,
        description="Minimum confidence for synthesis/summarize to bypass verify; below this fabrication is likely",
        ge=0.0,
        le=1.0
    )
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate graph type is one of the supported types."""
        valid_types = {"rag", "multi_turn"}
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {v}")
        return v


class Settings(BaseSettings):
    """Main settings class that aggregates all configuration.
    
    Configuration is loaded in the following order (later overrides earlier):
    1. Default values defined in this class
    2. YAML configuration file (configs/default.yaml)
    3. Environment variables
    
    Environment variables follow the pattern: SECTION_KEY (e.g., LLM_MODEL_NAME)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Nested configuration sections
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    
    # Global settings
    config_path: str = Field(
        default="configs/default.yaml",
        description="Path to YAML configuration file"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    api_key: str | None = Field(
        default=None,
        description="Bearer token for API authentication (GRAPHRAG_API_KEY). "
                    "If unset, the API is open — set this in production.",
    )
    
    def __init__(self, **data: Any) -> None:
        """Initialize settings, loading from YAML file first if it exists."""
        # Load YAML config first
        config_path = data.get("config_path", "configs/default.yaml")
        yaml_config = self._load_yaml_config(config_path)
        
        # Apply environment variable overrides to YAML config
        yaml_config = self._apply_env_overrides(yaml_config)
        
        # Merge YAML config with provided data (data takes precedence)
        merged = self._deep_merge(yaml_config, data)
        super().__init__(**merged)
    
    @staticmethod
    def _load_yaml_config(config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Dictionary with configuration values, or empty dict if file not found.
        """
        path = Path(config_path)
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration.
        
        Args:
            config: Configuration dictionary from YAML.
            
        Returns:
            Configuration with environment variable overrides applied.
        """
        # Define the mapping of env vars to nested config paths
        env_mappings = {
            # Corpus
            "CORPUS_NAME": ("corpus", "name"),
            "CORPUS_PATH": ("corpus", "path"),
            "CORPUS_CONNECTOR_TYPE": ("corpus", "connector_type"),
            # VectorStore
            "VECTORSTORE_TYPE": ("vectorstore", "type"),
            "VECTORSTORE_PERSIST_DIRECTORY": ("vectorstore", "persist_directory"),
            "VECTORSTORE_COLLECTION_NAME": ("vectorstore", "collection_name"),
            # Embeddings
            "EMBEDDINGS_PROVIDER": ("embeddings", "provider"),
            "EMBEDDINGS_MODEL_NAME": ("embeddings", "model_name"),
            "EMBEDDINGS_DIMENSION": ("embeddings", "dimension"),
            # LLM
            "LLM_PROVIDER": ("llm", "provider"),
            "LLM_MODEL_NAME": ("llm", "model_name"),
            "LLM_TEMPERATURE": ("llm", "temperature"),
            "LLM_MAX_TOKENS": ("llm", "max_tokens"),
            # Graph
            "GRAPH_TYPE": ("graph", "type"),
            "GRAPH_MAX_RETRIES": ("graph", "max_retries"),
            "GRAPH_REFUSAL_THRESHOLD": ("graph", "refusal_threshold"),
            # Global
            "DEBUG": ("debug",),
            "GRAPHRAG_API_KEY": ("api_key",),
        }
        
        result = config.copy()
        
        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Navigate to the nested location and set the value
                current = result
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = path[-1]
                current[final_key] = Settings._convert_env_value(value)
        
        return result
    
    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable.
            
        Returns:
            Converted value (int, float, bool, or str).
        """
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.
        
        Args:
            base: Base dictionary to merge into.
            override: Dictionary with values that override base.
            
        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def reload(self) -> None:
        """Reload configuration from the YAML file and environment variables."""
        yaml_config = self._load_yaml_config(self.config_path)
        yaml_config = self._apply_env_overrides(yaml_config)
        
        # Update nested configs
        if "corpus" in yaml_config:
            self.corpus = CorpusConfig(**yaml_config["corpus"])
        if "vectorstore" in yaml_config:
            self.vectorstore = VectorStoreConfig(**yaml_config["vectorstore"])
        if "embeddings" in yaml_config:
            self.embeddings = EmbeddingsConfig(**yaml_config["embeddings"])
        if "llm" in yaml_config:
            self.llm = LLMConfig(**yaml_config["llm"])
        if "graph" in yaml_config:
            self.graph = GraphConfig(**yaml_config["graph"])


# Global singleton instance
settings = Settings()
