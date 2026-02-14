"""Tests for structured logging infrastructure.

This module tests the logging configuration, JSON formatting,
and correlation ID support.
"""

import io
import json
import logging
import os
import sys
from unittest import mock

import pytest

from src import get_logger, get_correlation_id, set_correlation_id, clear_correlation_id
from src.logging_config import (
    JSONFormatter,
    configure_logging,
    get_log_level,
)


class TestLoggerCreation:
    """Test logger creation and configuration."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logging.Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_creates_logger_with_src_prefix(self):
        """Test that logger names are prefixed with 'src.'."""
        logger = get_logger("test_module")
        assert logger.name == "src.test_module"

    def test_get_logger_with_full_module_name(self):
        """Test get_logger with a full module name like __name__."""
        logger = get_logger("connectors.local")
        assert logger.name == "src.connectors.local"

    def test_get_logger_with_src_prefix_already_present(self):
        """Test that loggers already prefixed with 'src.' are not double-prefixed."""
        logger = get_logger("src.existing.module")
        assert logger.name == "src.existing.module"

    def test_get_logger_returns_same_logger_for_same_name(self):
        """Test that calling get_logger with the same name returns the same logger."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2


class TestLogLevelConfiguration:
    """Test log level configuration from environment."""

    def test_get_log_level_default(self):
        """Test default log level is INFO."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove LOG_LEVEL if present
            os.environ.pop("LOG_LEVEL", None)
            level = get_log_level()
            assert level == "INFO"

    def test_get_log_level_from_env(self):
        """Test log level is read from LOG_LEVEL environment variable."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            level = get_log_level()
            assert level == "DEBUG"

    def test_get_log_level_case_insensitive(self):
        """Test log level is case-insensitive."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
            level = get_log_level()
            assert level == "DEBUG"

    def test_get_log_level_invalid_falls_back_to_info(self):
        """Test invalid log level falls back to INFO."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            level = get_log_level()
            assert level == "INFO"

    def test_configure_logging_sets_level(self):
        """Test that configure_logging sets the correct level."""
        # Force reconfiguration
        configure_logging(level="DEBUG", force=True)
        
        src_logger = logging.getLogger("src")
        assert src_logger.level == logging.DEBUG

    def test_configure_logging_respects_env_var(self):
        """Test that configure_logging respects LOG_LEVEL env var."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            configure_logging(force=True)
            
            src_logger = logging.getLogger("src")
            assert src_logger.level == logging.WARNING


class TestJSONFormatOutput:
    """Test JSON formatting of log output."""

    def test_json_formatter_basic_fields(self):
        """Test that JSON formatter includes required fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="src.test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        log_entry = json.loads(output)
        
        assert "timestamp" in log_entry
        assert log_entry["level"] == "INFO"
        assert log_entry["module"] == "src.test.module"
        assert log_entry["message"] == "Test message"

    def test_json_formatter_timestamp_format(self):
        """Test that timestamp is in ISO 8601 format with Z suffix."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="src.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        log_entry = json.loads(output)
        
        # Timestamp should end with Z (UTC)
        assert log_entry["timestamp"].endswith("Z")
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(log_entry["timestamp"].replace("Z", "+00:00"))

    def test_json_formatter_extra_fields(self):
        """Test that extra fields are included in JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="src.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing document",
            args=(),
            exc_info=None,
        )
        # Add extra fields
        record.doc_id = "123"
        record.source = "s3"
        
        output = formatter.format(record)
        log_entry = json.loads(output)
        
        assert log_entry["doc_id"] == "123"
        assert log_entry["source"] == "s3"

    def test_json_formatter_exception_info(self):
        """Test that exception info is included in JSON output."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="src.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        
        output = formatter.format(record)
        log_entry = json.loads(output)
        
        assert "exception" in log_entry
        assert "ValueError: Test exception" in log_entry["exception"]

    def test_json_formatter_no_correlation_id_when_not_set(self):
        """Test that correlation_id is not present when not set."""
        # Clear any existing correlation ID
        clear_correlation_id()
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="src.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        log_entry = json.loads(output)
        
        assert "correlation_id" not in log_entry


class TestCorrelationIdPropagation:
    """Test correlation ID support for request tracing."""

    def test_set_correlation_id(self):
        """Test setting a correlation ID."""
        correlation_id = set_correlation_id()
        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        # Should be a UUID format
        assert len(correlation_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_set_correlation_id_with_custom_value(self):
        """Test setting a custom correlation ID."""
        custom_id = "custom-correlation-123"
        result = set_correlation_id(custom_id)
        assert result == custom_id
        assert get_correlation_id() == custom_id

    def test_get_correlation_id_returns_none_when_not_set(self):
        """Test that get_correlation_id returns None when not set."""
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_clear_correlation_id(self):
        """Test clearing the correlation ID."""
        set_correlation_id("test-id")
        assert get_correlation_id() == "test-id"
        
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_correlation_id_in_json_output(self):
        """Test that correlation ID is included in JSON log output."""
        clear_correlation_id()
        set_correlation_id("test-correlation-abc")
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="src.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        log_entry = json.loads(output)
        
        assert log_entry["correlation_id"] == "test-correlation-abc"
        
        # Cleanup
        clear_correlation_id()

    def test_correlation_id_propagates_to_multiple_logs(self):
        """Test that correlation ID propagates across multiple log calls."""
        clear_correlation_id()
        correlation_id = set_correlation_id("shared-correlation-id")
        
        formatter = JSONFormatter()
        
        # Create multiple log records
        records = [
            logging.LogRecord(
                name="src.module1",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="First message",
                args=(),
                exc_info=None,
            ),
            logging.LogRecord(
                name="src.module2",
                level=logging.INFO,
                pathname="test.py",
                lineno=2,
                msg="Second message",
                args=(),
                exc_info=None,
            ),
        ]
        
        outputs = [json.loads(formatter.format(r)) for r in records]
        
        # All outputs should have the same correlation ID
        assert outputs[0]["correlation_id"] == correlation_id
        assert outputs[1]["correlation_id"] == correlation_id
        
        # Cleanup
        clear_correlation_id()


class TestIntegration:
    """Integration tests for the logging system."""

    def test_full_logging_workflow(self, caplog):
        """Test the complete logging workflow with JSON output."""
        # Set up a string stream to capture output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        
        # Get logger and add our handler
        logger = get_logger("integration.test")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Set correlation ID
        set_correlation_id("integration-test-id")
        
        # Log a message with extra fields
        logger.info(
            "Processing document",
            extra={"doc_id": "doc-123", "source": "s3", "chunk_count": 5}
        )
        
        # Get the output
        output = stream.getvalue()
        log_entry = json.loads(output.strip())
        
        # Verify all expected fields
        assert log_entry["level"] == "INFO"
        assert log_entry["module"] == "src.integration.test"
        assert log_entry["message"] == "Processing document"
        assert log_entry["correlation_id"] == "integration-test-id"
        assert log_entry["doc_id"] == "doc-123"
        assert log_entry["source"] == "s3"
        assert log_entry["chunk_count"] == 5
        
        # Cleanup
        logger.removeHandler(handler)
        clear_correlation_id()

    def test_example_usage_from_task(self, capsys):
        """Test the example usage from the task description."""
        # Set up a custom handler to capture JSON output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        
        logger = get_logger("connectors.local")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Set a correlation ID
        set_correlation_id("abc-123")
        
        # Log as in the example
        logger.info("Processing document", extra={"doc_id": "123", "source": "s3"})
        
        # Parse output
        output = stream.getvalue()
        log_entry = json.loads(output.strip())
        
        # Verify output matches expected format
        assert "timestamp" in log_entry
        assert log_entry["level"] == "INFO"
        assert log_entry["module"] == "src.connectors.local"
        assert log_entry["message"] == "Processing document"
        assert log_entry["doc_id"] == "123"
        assert log_entry["source"] == "s3"
        assert log_entry["correlation_id"] == "abc-123"
        
        # Cleanup
        logger.removeHandler(handler)
        clear_correlation_id()


class TestModuleImport:
    """Test that the module can be imported correctly."""

    def test_import_from_src(self):
        """Test that get_logger can be imported from src."""
        from src import get_logger
        assert callable(get_logger)

    def test_import_correlation_functions(self):
        """Test that correlation ID functions can be imported from src."""
        from src import get_correlation_id, set_correlation_id, clear_correlation_id
        assert callable(get_correlation_id)
        assert callable(set_correlation_id)
        assert callable(clear_correlation_id)

    def test_logging_initialized_on_import(self):
        """Test that logging is initialized when src is imported."""
        # Re-import to ensure initialization
        import importlib
        import src
        importlib.reload(src)
        
        # Check that the src logger is configured
        src_logger = logging.getLogger("src")
        assert len(src_logger.handlers) > 0
        assert isinstance(src_logger.handlers[0].formatter, JSONFormatter)