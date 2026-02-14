"""Grounded GraphRAG Tutor service package.

This package provides structured logging through the get_logger function.
Logging is automatically configured on import with JSON formatting and
correlation ID support.

Usage:
    from src import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"doc_id": "123", "source": "s3"})
"""

from src.logging_config import (
    clear_correlation_id,
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)

# Initialize logging on import
configure_logging()

__all__ = [
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "configure_logging",
]
