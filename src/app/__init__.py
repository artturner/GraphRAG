"""FastAPI API layer for the Grounded GraphRAG Tutor service.

This module exports the FastAPI application instance for use in other modules.

Usage:
    from src.app import app
    
    # Run with: uvicorn src.app.main:app --reload
"""

from src.app.main import app, create_app

__all__ = ["app", "create_app"]
