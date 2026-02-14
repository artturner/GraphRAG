"""Document cleaning and chunking pipeline.

This module provides components for ingesting documents through
cleaning and chunking operations.

Example:
    ```python
    from src.ingestion import IngestionPipeline, TextCleaner, FixedSizeChunker
    from src.connectors import LocalConnector
    
    connector = LocalConnector("./data")
    pipeline = IngestionPipeline(
        connector=connector,
        cleaner=TextCleaner(),
        chunker=FixedSizeChunker(chunk_size=500)
    )
    result = pipeline.run()
    print(f"Ingested {result.documents_count} docs, {result.chunks_count} chunks")
    ```
"""

from src.ingestion.chunking import BaseChunker, FixedSizeChunker, SentenceChunker
from src.ingestion.cleaning import CleaningOptions, TextCleaner
from src.ingestion.pipeline import IngestProgress, IngestionPipeline

__all__ = [
    "IngestionPipeline",
    "IngestProgress",
    "TextCleaner",
    "CleaningOptions",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
]
