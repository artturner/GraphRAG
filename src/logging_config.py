"""Structured logging configuration for Grounded GraphRAG Tutor.

This module provides structured JSON logging with correlation ID support
for request tracing across the RAG pipeline.

Usage:
    from src import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"doc_id": "123", "source": "s3"})

Output format:
    {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "module": "connectors.local", 
     "message": "Processing document", "doc_id": "123", "source": "s3", 
     "correlation_id": "abc-123"}
"""

import json
import logging
import os
import sys
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


# Context variable for correlation ID tracking across async/threaded operations
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Thread-local storage for additional context
_context: threading.local = threading.local()

# Flag to track if logging has been initialized
_initialized = False


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context.
    
    Returns:
        The current correlation ID or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set a correlation ID in the current context.
    
    Args:
        correlation_id: Optional correlation ID to set. If None, generates a new UUID.
        
    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id.set(None)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging.
    
    Outputs log records as JSON with the following fields:
    - timestamp: ISO 8601 UTC timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - module: Module name where the log was emitted
    - message: Log message
    - correlation_id: Request correlation ID (if set)
    - Any extra fields passed to the log call
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Args:
            record: The log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        # Build the base log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if available
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add stack trace if present
        if record.stack_info:
            log_entry["stack_trace"] = self.formatStack(record.stack_info)
        
        # Add any extra fields from the record
        # These are attributes added via the extra parameter
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "asctime"
        }
        
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


def get_log_level() -> str:
    """Get the log level from environment variable or config.
    
    Returns:
        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Defaults to INFO.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    
    if level not in valid_levels:
        # Fall back to INFO if invalid level
        return "INFO"
    
    return level


def configure_logging(level: str | None = None, force: bool = False) -> None:
    """Configure structured JSON logging for the application.
    
    This should be called once at application startup. It configures
    the root logger to use JSON formatting.
    
    Args:
        level: Log level to use. If None, reads from LOG_LEVEL env var.
        force: If True, reconfigure logging even if already initialized.
    """
    global _initialized
    
    if _initialized and not force:
        return
    
    if level is None:
        level = get_log_level()
    
    # Get the root logger for the src package
    src_logger = logging.getLogger("src")
    
    # Clear any existing handlers
    src_logger.handlers.clear()
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    
    # Add handler and set level
    src_logger.addHandler(console_handler)
    src_logger.setLevel(level)
    
    # Prevent propagation to root logger to avoid duplicate logs
    src_logger.propagate = False
    
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.
    
    This function ensures consistent logger creation across the application.
    Loggers are created under the 'src' namespace for consistent configuration.
    
    Args:
        name: Module name, typically __name__.
        
    Returns:
        Configured logger instance.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"doc_id": "123"})
    """
    # Ensure logging is configured
    if not _initialized:
        configure_logging()
    
    # If the name doesn't start with 'src.', prepend it
    # This ensures all loggers share the same configuration
    if not name.startswith("src."):
        logger_name = f"src.{name}" if not name.startswith("src") else name
    else:
        logger_name = name
    
    return logging.getLogger(logger_name)


# Initialize logging on module import
configure_logging()
