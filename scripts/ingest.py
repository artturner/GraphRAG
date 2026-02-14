#!/usr/bin/env python3
"""
Ingestion script for the Grounded GraphRAG Tutor service.

This script loads documents from configured sources, processes them through
the ingestion pipeline (cleaning and chunking), and outputs progress and summary.

Usage:
    python scripts/ingest.py --corpus ./data
    python scripts/ingest.py --config configs/default.yaml
    python scripts/ingest.py --corpus ./data --chunk-size 500 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Settings
from src.connectors.factory import ConnectorFactory
from src.ingestion import (
    FixedSizeChunker,
    IngestProgress,
    IngestionPipeline,
    SentenceChunker,
    TextCleaner,
)

# Configure logging
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ingestion script.
    
    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into the GraphRAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest from local directory
    python scripts/ingest.py --corpus ./data
    
    # Use custom configuration
    python scripts/ingest.py --config configs/default.yaml
    
    # Override chunk size with verbose output
    python scripts/ingest.py --corpus ./data --chunk-size 500 --verbose
    
    # Use sentence-based chunking
    python scripts/ingest.py --corpus ./data --chunker sentence
        """,
    )
    
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to the document corpus directory (overrides config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for fixed-size chunker (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Overlap between chunks (default: 0)",
    )
    parser.add_argument(
        "--chunker",
        type=str,
        choices=["fixed", "sentence"],
        default="fixed",
        help="Chunking strategy to use (default: fixed)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=200,
        help="Minimum chunk size for sentence chunker (default: 200)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="Maximum chunk size for sentence chunker (default: 1000)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output (only show final summary)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )
    
    return parser


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity setting.
    
    Args:
        verbose: Whether to enable verbose logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_progress(progress: IngestProgress) -> None:
    """Print progress information to console.
    
    Args:
        progress: The current ingestion progress.
    """
    current = progress.current_file or "N/A"
    print(
        f"\rProgress: {progress.documents_processed} docs, "
        f"{progress.chunks_created} chunks | Current: {current[:50]}",
        end="",
        flush=True,
    )


def print_summary(result: IngestProgress, verbose: bool = False) -> None:
    """Print final summary of ingestion.
    
    Args:
        result: The final ingestion result.
        verbose: Whether to show detailed information.
    """
    print("\n")  # New line after progress
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents processed: {result.documents_processed}")
    print(f"Chunks created:      {result.chunks_created}")
    
    if result.errors:
        print(f"\nErrors encountered:  {len(result.errors)}")
        if verbose:
            for error in result.errors:
                print(f"  - {error}")
    else:
        print("\nNo errors encountered.")
    
    print("=" * 60)


def main() -> int:
    """Run the ingestion pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    settings = Settings(config_path=args.config)
    
    # Determine corpus path
    corpus_path = args.corpus or settings.corpus.path
    logger.info(f"Using corpus path: {corpus_path}")
    
    # Validate corpus path exists
    if not Path(corpus_path).exists():
        logger.error(f"Corpus path does not exist: {corpus_path}")
        print(f"Error: Corpus path does not exist: {corpus_path}")
        return 1
    
    try:
        # Create connector
        logger.info(f"Creating connector for type: {settings.corpus.connector_type}")
        connector = ConnectorFactory.create(
            connector_type=settings.corpus.connector_type,
            source=corpus_path,
        )
        
        # Validate connector
        if not connector.validate_source():
            logger.error(f"Invalid source: {corpus_path}")
            print(f"Error: Invalid source: {corpus_path}")
            return 1
        
        # Create cleaner
        cleaner = TextCleaner()
        logger.debug("Created TextCleaner with default options")
        
        # Create chunker based on type
        if args.chunker == "sentence":
            chunker = SentenceChunker(
                min_size=args.min_size,
                max_size=args.max_size,
            )
            logger.debug(
                f"Created SentenceChunker (min={args.min_size}, max={args.max_size})"
            )
        else:
            chunker = FixedSizeChunker(
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
            logger.debug(
                f"Created FixedSizeChunker (size={args.chunk_size}, overlap={args.chunk_overlap})"
            )
        
        # Create pipeline
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )
        logger.info(f"Created pipeline: {pipeline}")
        
        # Run pipeline
        print(f"Starting ingestion from: {corpus_path}")
        print(f"Chunker: {args.chunker} (size={args.chunk_size if args.chunker == 'fixed' else 'N/A'})")
        print("-" * 60)
        
        if args.no_progress:
            # Run without progress reporting
            result = pipeline.run()
            final_progress = IngestProgress(
                documents_processed=result.documents_count,
                chunks_created=result.chunks_count,
                errors=result.errors,
            )
        else:
            # Run with progress reporting
            final_progress = IngestProgress()
            for progress in pipeline.run_with_progress():
                print_progress(progress)
                final_progress = progress
        
        # Print summary
        print_summary(final_progress, args.verbose)
        
        # Return success if no errors, or partial success
        if final_progress.errors:
            return 1 if final_progress.documents_processed == 0 else 0
        return 0
        
    except Exception as e:
        logger.exception("Ingestion failed with error")
        print(f"\nError: Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
