"""Factory for creating vector store instances.

This module provides a factory class for creating vector store instances
based on configuration. It supports FAISS and ChromaDB backends.

Example:
    ```python
    from src.store import VectorStoreFactory
    from src.config import VectorStoreConfig
    
    config = VectorStoreConfig(type="faiss", persist_directory="./data/index")
    store = VectorStoreFactory.get_store(config, dimension=384)
    ```
"""

import logging
from typing import Type

from src.config import VectorStoreConfig
from src.exceptions import ConfigurationError
from src.store.base import BaseVectorStore
from src.store.faiss_store import FAISSVectorStore
from src.store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances.
    
    This factory class provides a centralized way to create vector store
    instances based on configuration. It abstracts away the details of
    instantiating different store types.
    
    Supported store types:
        - "faiss": FAISS-based in-memory vector store with optional persistence
        - "chroma": ChromaDB-based vector store with persistent storage
    
    Example:
        ```python
        # Create FAISS store
        config = VectorStoreConfig(type="faiss", persist_directory="./data/index")
        store = VectorStoreFactory.get_store(config, dimension=384)
        
        # Create ChromaDB store
        config = VectorStoreConfig(
            type="chroma",
            persist_directory="./data/chroma",
            collection_name="documents"
        )
        store = VectorStoreFactory.get_store(config, dimension=384)
        ```
    """
    
    # Mapping of store type names to their implementation classes
    _STORES: dict[str, Type[BaseVectorStore]] = {
        "faiss": FAISSVectorStore,
        "chroma": ChromaVectorStore,
    }
    
    @classmethod
    def get_store(
        cls,
        config: VectorStoreConfig,
        dimension: int,
    ) -> BaseVectorStore:
        """Create a vector store instance based on configuration.
        
        Args:
            config: VectorStoreConfig instance specifying the store type and settings.
            dimension: The dimensionality of embedding vectors.
            
        Returns:
            An instance of the appropriate BaseVectorStore subclass.
            
        Raises:
            ConfigurationError: If the store type is not supported or
                configuration is invalid.
        
        Example:
            ```python
            config = VectorStoreConfig(type="faiss", persist_directory="./data/index")
            store = VectorStoreFactory.get_store(config, dimension=384)
            print(f"Store type: {type(store).__name__}")  # FAISSVectorStore
            ```
        """
        store_type = config.type
        
        # Check if store type is supported
        if store_type not in cls._STORES:
            valid_types = sorted(cls._STORES.keys())
            raise ConfigurationError(
                f"Unsupported vector store type: '{store_type}'. "
                f"Valid types are: {valid_types}"
            )
        
        # Get the store class
        store_class = cls._STORES[store_type]
        
        # Create instance based on store type
        logger.info(
            f"Creating vector store: {store_type} with dimension: {dimension}"
        )
        
        if store_type == "faiss":
            return cls._create_faiss_store(config, store_class, dimension)
        elif store_type == "chroma":
            return cls._create_chroma_store(config, store_class)
        else:
            # This should not happen due to the check above, but just in case
            raise ConfigurationError(f"Unknown store type: {store_type}")
    
    @classmethod
    def _create_faiss_store(
        cls,
        config: VectorStoreConfig,
        store_class: Type[FAISSVectorStore],
        dimension: int,
    ) -> FAISSVectorStore:
        """Create a FAISSVectorStore instance.
        
        Args:
            config: VectorStoreConfig with store settings.
            store_class: The FAISSVectorStore class to instantiate.
            dimension: The dimensionality of embedding vectors.
            
        Returns:
            A configured FAISSVectorStore instance.
        """
        return store_class(
            dimension=dimension,
            persist_dir=config.persist_directory,
        )
    
    @classmethod
    def _create_chroma_store(
        cls,
        config: VectorStoreConfig,
        store_class: Type[ChromaVectorStore],
    ) -> ChromaVectorStore:
        """Create a ChromaVectorStore instance.
        
        Args:
            config: VectorStoreConfig with store settings.
            store_class: The ChromaVectorStore class to instantiate.
            
        Returns:
            A configured ChromaVectorStore instance.
        """
        return store_class(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )
    
    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get a list of supported store type names.
        
        Returns:
            A sorted list of supported store type names.
        
        Example:
            ```python
            types = VectorStoreFactory.get_supported_types()
            print(types)  # ['chroma', 'faiss']
            ```
        """
        return sorted(cls._STORES.keys())
    
    @classmethod
    def is_type_supported(cls, store_type: str) -> bool:
        """Check if a store type is supported.
        
        Args:
            store_type: The store type name to check.
            
        Returns:
            True if the store type is supported, False otherwise.
        
        Example:
            ```python
            if VectorStoreFactory.is_type_supported("faiss"):
                config = VectorStoreConfig(type="faiss")
                store = VectorStoreFactory.get_store(config, dimension=384)
            ```
        """
        return store_type in cls._STORES
    
    @classmethod
    def register_store(
        cls,
        store_type: str,
        store_class: Type[BaseVectorStore],
    ) -> None:
        """Register a new store type.
        
        This allows for dynamic extension of the factory with custom
        store implementations.
        
        Args:
            store_type: The name to register the store under.
            store_class: The store class to register.
        
        Raises:
            TypeError: If store_class is not a subclass of BaseVectorStore.
        
        Example:
            ```python
            from src.store import VectorStoreFactory, BaseVectorStore
            
            class MyVectorStore(BaseVectorStore):
                # Implementation...
                pass
            
            VectorStoreFactory.register_store("my_store", MyVectorStore)
            ```
        """
        if not isinstance(store_class, type) or not issubclass(
            store_class, BaseVectorStore
        ):
            raise TypeError(
                f"store_class must be a subclass of BaseVectorStore, "
                f"got {type(store_class)}"
            )
        
        cls._STORES[store_type] = store_class
        logger.info(f"Registered vector store type: {store_type}")
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered stores (useful for testing).
        
        Note:
            This will remove all stores including the default ones.
            Use with caution, primarily for testing purposes.
        """
        cls._STORES.clear()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the factory to default store registrations.
        
        This is useful after testing with custom stores to restore
        the default store types.
        """
        cls._STORES = {
            "faiss": FAISSVectorStore,
            "chroma": ChromaVectorStore,
        }