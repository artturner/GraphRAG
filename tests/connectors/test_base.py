"""Tests for the base connector interface.

This module tests the abstract BaseConnector class and ConnectorRegistry
to ensure proper behavior for connector implementations.
"""

import tempfile
from pathlib import Path
from typing import Type

import pytest

from src.connectors import BaseConnector, ConnectorRegistry
from src.types import Document


class TestBaseConnector:
    """Tests for the BaseConnector abstract class."""
    
    def test_cannot_instantiate_base_connector_directly(self) -> None:
        """Test that BaseConnector cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseConnector("/some/path")  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_subclass_must_implement_load(self) -> None:
        """Test that subclass must implement the load method."""
        class IncompleteConnector(BaseConnector):
            def list_documents(self) -> list[str]:
                return []
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteConnector("/some/path")  # type: ignore[abstract]
        
        assert "load" in str(exc_info.value).lower()
    
    def test_subclass_must_implement_list_documents(self) -> None:
        """Test that subclass must implement the list_documents method."""
        class IncompleteConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteConnector("/some/path")  # type: ignore[abstract]
        
        assert "list_documents" in str(exc_info.value).lower()
    
    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that a complete subclass can be instantiated."""
        class CompleteConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        connector = CompleteConnector("/some/path")
        assert connector.source == "/some/path"
    
    def test_source_property_returns_source(self) -> None:
        """Test that the source property returns the configured source."""
        class TestConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        connector = TestConnector("/test/source/path")
        assert connector.source == "/test/source/path"
    
    def test_validate_source_returns_true_for_existing_path(self) -> None:
        """Test that validate_source returns True for existing paths."""
        class TestConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            connector = TestConnector(tmpdir)
            assert connector.validate_source() is True
    
    def test_validate_source_returns_false_for_nonexistent_path(self) -> None:
        """Test that validate_source returns False for non-existent paths."""
        class TestConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        connector = TestConnector("/nonexistent/path/that/does/not/exist")
        assert connector.validate_source() is False
    
    def test_subclass_can_override_validate_source(self) -> None:
        """Test that subclasses can override validate_source."""
        class CustomConnector(BaseConnector):
            def __init__(self, source: str, should_validate: bool) -> None:
                super().__init__(source)
                self._should_validate = should_validate
            
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
            
            def validate_source(self) -> bool:
                return self._should_validate
        
        connector = CustomConnector("/any/path", should_validate=True)
        assert connector.validate_source() is True
        
        connector2 = CustomConnector("/any/path", should_validate=False)
        assert connector2.validate_source() is False


class TestConnectorRegistry:
    """Tests for the ConnectorRegistry class."""
    
    def setup_method(self) -> None:
        """Clear the registry before each test."""
        ConnectorRegistry.clear()
    
    def test_register_connector_with_decorator(self) -> None:
        """Test registering a connector using the decorator."""
        @ConnectorRegistry.register("test_connector")
        class TestConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        assert ConnectorRegistry.get("test_connector") is TestConnector
    
    def test_register_connector_explicitly(self) -> None:
        """Test registering a connector explicitly."""
        class TestConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            
            def list_documents(self) -> list[str]:
                return []
        
        ConnectorRegistry.register_connector("explicit_connector", TestConnector)
        assert ConnectorRegistry.get("explicit_connector") is TestConnector
    
    def test_register_non_connector_raises_error(self) -> None:
        """Test that registering a non-BaseConnector class raises TypeError."""
        class NotAConnector:
            pass
        
        with pytest.raises(TypeError) as exc_info:
            ConnectorRegistry.register_connector("invalid", NotAConnector)  # type: ignore[type-var]
        
        assert "BaseConnector" in str(exc_info.value)
    
    def test_get_nonexistent_connector_returns_none(self) -> None:
        """Test that getting a non-existent connector returns None."""
        result = ConnectorRegistry.get("nonexistent")
        assert result is None
    
    def test_list_connectors_returns_all_names(self) -> None:
        """Test that list_connectors returns all registered connector names."""
        @ConnectorRegistry.register("connector_a")
        class ConnectorA(BaseConnector):
            def load(self) -> list[Document]:
                return []
            def list_documents(self) -> list[str]:
                return []
        
        @ConnectorRegistry.register("connector_b")
        class ConnectorB(BaseConnector):
            def load(self) -> list[Document]:
                return []
            def list_documents(self) -> list[str]:
                return []
        
        names = ConnectorRegistry.list_connectors()
        assert set(names) == {"connector_a", "connector_b"}
    
    def test_list_connectors_empty_when_cleared(self) -> None:
        """Test that list_connectors returns empty list when cleared."""
        @ConnectorRegistry.register("temp_connector")
        class TempConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            def list_documents(self) -> list[str]:
                return []
        
        ConnectorRegistry.clear()
        assert ConnectorRegistry.list_connectors() == []
    
    def test_clear_removes_all_connectors(self) -> None:
        """Test that clear removes all registered connectors."""
        @ConnectorRegistry.register("to_clear")
        class ToClearConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            def list_documents(self) -> list[str]:
                return []
        
        assert ConnectorRegistry.get("to_clear") is not None
        ConnectorRegistry.clear()
        assert ConnectorRegistry.get("to_clear") is None
    
    def test_register_overwrites_existing_connector(self) -> None:
        """Test that registering with the same name overwrites the previous."""
        @ConnectorRegistry.register("duplicate")
        class FirstConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            def list_documents(self) -> list[str]:
                return []
        
        @ConnectorRegistry.register("duplicate")
        class SecondConnector(BaseConnector):
            def load(self) -> list[Document]:
                return []
            def list_documents(self) -> list[str]:
                return []
        
        assert ConnectorRegistry.get("duplicate") is SecondConnector
        assert ConnectorRegistry.get("duplicate") is not FirstConnector


class TestConnectorImplementation:
    """Tests for a complete connector implementation."""
    
    def test_full_connector_implementation(self, tmp_path: Path) -> None:
        """Test a full connector implementation with real files."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Content of document 1")
        (tmp_path / "doc2.txt").write_text("Content of document 2")
        
        class FileConnector(BaseConnector):
            """Simple file connector for testing."""
            
            def load(self) -> list[Document]:
                documents = []
                source_path = Path(self.source)
                for file_path in source_path.glob("*.txt"):
                    doc = Document(
                        id=file_path.stem,
                        content=file_path.read_text(),
                        source=str(file_path),
                        metadata={"filename": file_path.name}
                    )
                    documents.append(doc)
                return documents
            
            def list_documents(self) -> list[str]:
                source_path = Path(self.source)
                return [f.stem for f in source_path.glob("*.txt")]
        
        connector = FileConnector(str(tmp_path))
        
        # Test validate_source
        assert connector.validate_source() is True
        
        # Test list_documents
        doc_names = sorted(connector.list_documents())
        assert doc_names == ["doc1", "doc2"]
        
        # Test load
        documents = sorted(connector.load(), key=lambda d: d.id)
        assert len(documents) == 2
        assert documents[0].id == "doc1"
        assert documents[0].content == "Content of document 1"
        assert documents[1].id == "doc2"
        assert documents[1].content == "Content of document 2"
