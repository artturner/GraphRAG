"""Document source adapters for loading documents from various sources.

This module provides the base connector interface, a registry for
managing connector implementations, and a factory for instantiating
connectors based on configuration.

Example:
    ```python
    from src.connectors import ConnectorFactory, BaseConnector
    from src.config import CorpusConfig
    
    # Create connector from configuration
    config = CorpusConfig(connector_type="local", path="./data")
    connector = ConnectorFactory.get_connector(config)
    documents = connector.load()
    
    # Or use the registry for custom connectors
    from src.connectors import ConnectorRegistry
    
    @ConnectorRegistry.register("my_source")
    class MyConnector(BaseConnector):
        def load(self) -> list[Document]:
            pass
        
        def list_documents(self) -> list[str]:
            pass
    ```
"""

from typing import Optional, Type

from src.connectors.base import BaseConnector
from src.connectors.factory import ConnectorFactory
from src.connectors.local import LocalConnector
from src.connectors.s3 import S3Connector
from src.connectors.web import WebConnector


class ConnectorRegistry:
    """Registry for document connector implementations.
    
    This class provides a central registry for managing connector implementations,
    allowing connectors to be registered and retrieved by name.
    
    The registry supports both decorator-based registration and explicit
    registration of connector classes.
    
    Example:
        ```python
        # Register using decorator
        @ConnectorRegistry.register("file")
        class FileConnector(BaseConnector):
            pass
        
        # Register explicitly
        ConnectorRegistry.register_connector("database", DatabaseConnector)
        
        # Get a connector class
        connector_class = ConnectorRegistry.get("file")
        ```
    """
    
    _connectors: dict[str, Type[BaseConnector]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a connector class.
        
        Args:
            name: The name to register the connector under.
        
        Returns:
            A decorator function that registers the connector.
        
        Example:
            ```python
            @ConnectorRegistry.register("my_connector")
            class MyConnector(BaseConnector):
                pass
            ```
        """
        def decorator(connector_class: Type[BaseConnector]) -> Type[BaseConnector]:
            cls._connectors[name] = connector_class
            return connector_class
        return decorator
    
    @classmethod
    def register_connector(cls, name: str, connector_class: Type[BaseConnector]) -> None:
        """Explicitly register a connector class.
        
        Args:
            name: The name to register the connector under.
            connector_class: The connector class to register.
        
        Raises:
            TypeError: If the connector_class is not a subclass of BaseConnector.
        
        Example:
            ```python
            ConnectorRegistry.register_connector("file", FileConnector)
            ```
        """
        if not isinstance(connector_class, type) or not issubclass(connector_class, BaseConnector):
            raise TypeError(
                f"connector_class must be a subclass of BaseConnector, "
                f"got {type(connector_class)}"
            )
        cls._connectors[name] = connector_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseConnector]]:
        """Get a connector class by name.
        
        Args:
            name: The name of the connector to retrieve.
        
        Returns:
            The connector class if found, None otherwise.
        
        Example:
            ```python
            connector_class = ConnectorRegistry.get("file")
            if connector_class:
                connector = connector_class("/path/to/files")
            ```
        """
        return cls._connectors.get(name)
    
    @classmethod
    def list_connectors(cls) -> list[str]:
        """List all registered connector names.
        
        Returns:
            A list of registered connector names.
        
        Example:
            ```python
            names = ConnectorRegistry.list_connectors()
            print(f"Available connectors: {names}")
            ```
        """
        return list(cls._connectors.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered connectors.
        
        This is primarily useful for testing purposes.
        
        Example:
            ```python
            ConnectorRegistry.clear()
            ```
        """
        cls._connectors.clear()


__all__ = [
    "BaseConnector",
    "ConnectorFactory",
    "ConnectorRegistry",
    "LocalConnector",
    "S3Connector",
    "WebConnector",
]
