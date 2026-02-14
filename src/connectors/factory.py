"""Factory for creating document connectors based on configuration.

This module provides a factory class for instantiating the appropriate
connector based on configuration settings.
"""

import logging
from typing import Type

from src.config import CorpusConfig
from src.connectors.base import BaseConnector
from src.connectors.local import LocalConnector
from src.connectors.s3 import S3Connector
from src.connectors.web import WebConnector
from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Factory for creating document connectors based on configuration.
    
    This factory provides a centralized way to instantiate connectors
    based on the connector type specified in the configuration. It supports
    dynamic registration of new connector types.
    
    Supported connector types:
        - "local": LocalConnector for loading from local filesystem
        - "s3": S3Connector for loading from AWS S3 (stub)
        - "web": WebConnector for loading from web URLs (stub)
    
    Example:
        ```python
        from src.connectors import ConnectorFactory
        from src.config import CorpusConfig
        
        # Create connector from configuration
        config = CorpusConfig(connector_type="local", path="./data")
        connector = ConnectorFactory.get_connector(config)
        documents = connector.load()
        
        # Register a custom connector
        class MyConnector(BaseConnector):
            pass
        
        ConnectorFactory.register_connector("my_type", MyConnector)
        config = CorpusConfig(connector_type="my_type", path="./data")
        connector = ConnectorFactory.get_connector(config)
        ```
    """
    
    # Registry of connector types to their classes
    _connectors: dict[str, Type[BaseConnector]] = {
        "local": LocalConnector,
        "s3": S3Connector,
        "web": WebConnector,
    }
    
    @classmethod
    def get_connector(cls, config: CorpusConfig) -> BaseConnector:
        """Create and return a connector based on the configuration.
        
        Args:
            config: CorpusConfig containing connector_type and path settings.
        
        Returns:
            An instance of the appropriate connector for the configuration.
        
        Raises:
            ConfigurationError: If the connector type is not recognized.
        
        Example:
            ```python
            from src.connectors import ConnectorFactory
            from src.config import CorpusConfig
            
            config = CorpusConfig(connector_type="local", path="./data")
            connector = ConnectorFactory.get_connector(config)
            ```
        """
        connector_type = config.connector_type
        
        if connector_type not in cls._connectors:
            raise ConfigurationError(
                f"Unknown connector type: '{connector_type}'",
                details=f"Supported types: {list(cls._connectors.keys())}"
            )
        
        connector_class = cls._connectors[connector_type]
        
        # Create connector instance based on type
        if connector_type == "local":
            return connector_class(source_path=config.path)
        elif connector_type == "s3":
            return connector_class(source=config.path)
        elif connector_type == "web":
            return connector_class(source=config.path)
        else:
            # For custom connectors, try to pass path as source
            return connector_class(source=config.path)
    
    @classmethod
    def register_connector(
        cls,
        connector_type: str,
        connector_class: Type[BaseConnector],
    ) -> None:
        """Register a new connector type.
        
        This allows for dynamic extension of the factory with custom
        connector implementations.
        
        Args:
            connector_type: The name to register the connector under.
            connector_class: The connector class to register.
        
        Raises:
            TypeError: If connector_class is not a subclass of BaseConnector.
        
        Example:
            ```python
            from src.connectors import ConnectorFactory, BaseConnector
            
            class DatabaseConnector(BaseConnector):
                def load(self) -> list[Document]:
                    pass
                
                def list_documents(self) -> list[str]:
                    pass
            
            ConnectorFactory.register_connector("database", DatabaseConnector)
            ```
        """
        if not isinstance(connector_class, type) or not issubclass(
            connector_class, BaseConnector
        ):
            raise TypeError(
                f"connector_class must be a subclass of BaseConnector, "
                f"got {type(connector_class)}"
            )
        
        cls._connectors[connector_type] = connector_class
        logger.info(f"Registered connector type: {connector_type}")
    
    @classmethod
    def list_connectors(cls) -> list[str]:
        """List all registered connector types.
        
        Returns:
            A list of registered connector type names.
        
        Example:
            ```python
            types = ConnectorFactory.list_connectors()
            print(f"Available connector types: {types}")
            # Output: ['local', 's3', 'web']
            ```
        """
        return list(cls._connectors.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered connectors (useful for testing).
        
        Note:
            This will remove all connectors including the default ones.
            Use with caution, primarily for testing purposes.
        
        Example:
            ```python
            ConnectorFactory.clear()
            # Now re-register only the connectors you need
            ConnectorFactory.register_connector("local", LocalConnector)
            ```
        """
        cls._connectors.clear()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the factory to default connector registrations.
        
        This is useful after testing with custom connectors to restore
        the default connector types.
        
        Example:
            ```python
            # After custom testing
            ConnectorFactory.reset()
            # Factory now has default connectors: local, s3, web
            ```
        """
        cls._connectors = {
            "local": LocalConnector,
            "s3": S3Connector,
            "web": WebConnector,
        }