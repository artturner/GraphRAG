"""Tests for the ConnectorFactory.

This module tests the factory for instantiating connectors based on configuration.
"""

import pytest

from src.config import CorpusConfig
from src.connectors import (
    BaseConnector,
    ConnectorFactory,
    LocalConnector,
    S3Connector,
    WebConnector,
)
from src.exceptions import ConfigurationError


class TestConnectorFactory:
    """Test suite for ConnectorFactory class."""
    
    def test_get_connector_returns_local_connector(self) -> None:
        """Test that factory returns LocalConnector for 'local' type."""
        config = CorpusConfig(connector_type="local", path="./data")
        connector = ConnectorFactory.get_connector(config)
        
        assert isinstance(connector, LocalConnector)
        assert isinstance(connector, BaseConnector)
    
    def test_get_connector_returns_s3_connector(self) -> None:
        """Test that factory returns S3Connector for 's3' type."""
        config = CorpusConfig(connector_type="s3", path="s3://my-bucket")
        connector = ConnectorFactory.get_connector(config)
        
        assert isinstance(connector, S3Connector)
        assert isinstance(connector, BaseConnector)
    
    def test_get_connector_returns_web_connector(self) -> None:
        """Test that factory returns WebConnector for 'web' type."""
        config = CorpusConfig(connector_type="web", path="https://example.com")
        connector = ConnectorFactory.get_connector(config)
        
        assert isinstance(connector, WebConnector)
        assert isinstance(connector, BaseConnector)
    
    def test_get_connector_raises_error_for_unknown_type(self) -> None:
        """Test that factory raises ConfigurationError for unknown type."""
        # Create config with invalid type by bypassing validation
        config = CorpusConfig(connector_type="local", path="./data")
        # Manually set an invalid type to test error handling
        config.connector_type = "unknown_type"
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConnectorFactory.get_connector(config)
        
        assert "Unknown connector type" in str(exc_info.value)
        assert "unknown_type" in str(exc_info.value)
    
    def test_get_connector_with_local_config_uses_path(self) -> None:
        """Test that local connector is created with the correct path."""
        config = CorpusConfig(connector_type="local", path="./test_documents")
        connector = ConnectorFactory.get_connector(config)
        
        assert connector.source_path.name == "test_documents"
    
    def test_get_connector_with_s3_config_uses_source(self) -> None:
        """Test that S3 connector is created with the correct source."""
        config = CorpusConfig(connector_type="s3", path="s3://my-bucket/docs")
        connector = ConnectorFactory.get_connector(config)
        
        assert connector.source == "s3://my-bucket/docs"
    
    def test_get_connector_with_web_config_uses_source(self) -> None:
        """Test that web connector is created with the correct source."""
        config = CorpusConfig(
            connector_type="web",
            path="https://example.com/documents/"
        )
        connector = ConnectorFactory.get_connector(config)
        
        assert connector.source == "https://example.com/documents/"
    
    def test_list_connectors_returns_all_types(self) -> None:
        """Test that list_connectors returns all registered types."""
        connectors = ConnectorFactory.list_connectors()
        
        assert "local" in connectors
        assert "s3" in connectors
        assert "web" in connectors
    
    def test_register_connector_adds_new_type(self) -> None:
        """Test that register_connector adds a new connector type."""
        # Create a custom connector class
        class CustomConnector(BaseConnector):
            def __init__(self, source: str) -> None:
                super().__init__(source)
            
            def load(self) -> list:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        # Register the custom connector
        ConnectorFactory.register_connector("custom", CustomConnector)
        
        try:
            # Verify it's registered
            assert "custom" in ConnectorFactory.list_connectors()
            
            # Create config and get connector
            config = CorpusConfig(connector_type="local", path="./data")
            config.connector_type = "custom"
            connector = ConnectorFactory.get_connector(config)
            
            assert isinstance(connector, CustomConnector)
        finally:
            # Clean up - reset to defaults
            ConnectorFactory.reset()
    
    def test_register_connector_raises_for_non_connector_class(self) -> None:
        """Test that register_connector raises TypeError for invalid class."""
        with pytest.raises(TypeError) as exc_info:
            ConnectorFactory.register_connector("invalid", str)  # type: ignore
        
        assert "must be a subclass of BaseConnector" in str(exc_info.value)
    
    def test_register_connector_raises_for_instance(self) -> None:
        """Test that register_connector raises TypeError for instance."""
        with pytest.raises(TypeError) as exc_info:
            ConnectorFactory.register_connector("invalid", "not a class")  # type: ignore
        
        assert "must be a subclass of BaseConnector" in str(exc_info.value)
    
    def test_reset_restores_default_connectors(self) -> None:
        """Test that reset restores the default connector types."""
        # Clear all connectors
        ConnectorFactory.clear()
        assert ConnectorFactory.list_connectors() == []
        
        # Reset to defaults
        ConnectorFactory.reset()
        
        connectors = ConnectorFactory.list_connectors()
        assert "local" in connectors
        assert "s3" in connectors
        assert "web" in connectors
    
    def test_clear_removes_all_connectors(self) -> None:
        """Test that clear removes all registered connectors."""
        # Ensure we have default connectors
        assert len(ConnectorFactory.list_connectors()) > 0
        
        try:
            ConnectorFactory.clear()
            assert ConnectorFactory.list_connectors() == []
        finally:
            # Restore defaults
            ConnectorFactory.reset()


class TestS3ConnectorStub:
    """Test suite for S3Connector stub implementation."""
    
    def test_s3_connector_load_raises_not_implemented(self) -> None:
        """Test that S3Connector.load() raises NotImplementedError."""
        connector = S3Connector(source="s3://my-bucket")
        
        with pytest.raises(NotImplementedError) as exc_info:
            connector.load()
        
        assert "not yet implemented" in str(exc_info.value).lower()
    
    def test_s3_connector_list_documents_raises_not_implemented(self) -> None:
        """Test that S3Connector.list_documents() raises NotImplementedError."""
        connector = S3Connector(source="s3://my-bucket")
        
        with pytest.raises(NotImplementedError) as exc_info:
            connector.list_documents()
        
        assert "not yet implemented" in str(exc_info.value).lower()
    
    def test_s3_connector_validate_source_raises_not_implemented(self) -> None:
        """Test that S3Connector.validate_source() raises NotImplementedError."""
        connector = S3Connector(source="s3://my-bucket")
        
        with pytest.raises(NotImplementedError) as exc_info:
            connector.validate_source()
        
        assert "not yet implemented" in str(exc_info.value).lower()
    
    def test_s3_connector_extracts_bucket_from_uri(self) -> None:
        """Test that S3Connector extracts bucket name from URI."""
        connector = S3Connector(source="s3://my-bucket/path/to/docs")
        
        assert connector.bucket == "my-bucket"
    
    def test_s3_connector_uses_explicit_bucket(self) -> None:
        """Test that S3Connector uses explicit bucket parameter."""
        connector = S3Connector(
            source="s3://other-bucket",
            bucket="my-explicit-bucket"
        )
        
        assert connector.bucket == "my-explicit-bucket"


class TestWebConnectorStub:
    """Test suite for WebConnector stub implementation."""
    
    def test_web_connector_load_raises_not_implemented(self) -> None:
        """Test that WebConnector.load() raises NotImplementedError."""
        connector = WebConnector(source="https://example.com")
        
        with pytest.raises(NotImplementedError) as exc_info:
            connector.load()
        
        assert "not yet implemented" in str(exc_info.value).lower()
    
    def test_web_connector_list_documents_raises_not_implemented(self) -> None:
        """Test that WebConnector.list_documents() raises NotImplementedError."""
        connector = WebConnector(source="https://example.com")
        
        with pytest.raises(NotImplementedError) as exc_info:
            connector.list_documents()
        
        assert "not yet implemented" in str(exc_info.value).lower()
    
    def test_web_connector_validate_source_raises_not_implemented(self) -> None:
        """Test that WebConnector.validate_source() raises NotImplementedError."""
        connector = WebConnector(source="https://example.com")
        
        with pytest.raises(NotImplementedError) as exc_info:
            connector.validate_source()
        
        assert "not yet implemented" in str(exc_info.value).lower()
    
    def test_web_connector_has_base_url(self) -> None:
        """Test that WebConnector stores base_url."""
        connector = WebConnector(source="https://example.com/docs")
        
        assert connector.base_url == "https://example.com/docs"
    
    def test_web_connector_has_timeout(self) -> None:
        """Test that WebConnector has configurable timeout."""
        connector = WebConnector(source="https://example.com", timeout=60)
        
        assert connector.timeout == 60


class TestConnectorFactoryIntegration:
    """Integration tests for ConnectorFactory with real configs."""
    
    def test_factory_with_local_connector_and_fixture_path(self) -> None:
        """Test factory with local connector using test fixtures path."""
        config = CorpusConfig(
            connector_type="local",
            path="tests/fixtures/documents"
        )
        connector = ConnectorFactory.get_connector(config)
        
        assert isinstance(connector, LocalConnector)
        assert connector.validate_source()
        
        # Should be able to list documents
        docs = connector.list_documents()
        assert len(docs) > 0
        assert any("sample" in doc.lower() for doc in docs)
    
    def test_factory_creates_new_instance_each_call(self) -> None:
        """Test that factory creates a new connector instance each call."""
        config = CorpusConfig(connector_type="local", path="./data")
        
        connector1 = ConnectorFactory.get_connector(config)
        connector2 = ConnectorFactory.get_connector(config)
        
        assert connector1 is not connector2