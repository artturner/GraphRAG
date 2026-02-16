#!/usr/bin/env python3
"""
Ingestion script for the Grounded GraphRAG Tutor service.

This script loads documents from configured sources, processes them through
the ingestion pipeline (cleaning and chunking), indexes them into the vector
store, and outputs progress and summary.

Usage:
    python scripts/ingest.py --corpus ./data
    python scripts/ingest.py --config configs/default.yaml
    python scripts/ingest.py --corpus ./data --chunk-size 500 --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path for imports
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ingestion script."""
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

    # Ingest and index into the vector store
    python scripts/ingest.py --corpus ./data --index
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
        "--index",
        action="store_true",
        help="Also index chunks into the vector store after ingestion",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output (only show final summary)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Return a text-based progress bar string."""
    if total <= 0:
        return f"[{'?' * width}]"
    ratio = min(current / total, 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def print_progress(progress: IngestProgress, total_docs: int | None = None) -> None:
    """Print progress information to console."""
    current = progress.current_file or "N/A"
    truncated = current[:50]
    if total_docs and total_docs > 0:
        bar = _progress_bar(progress.documents_processed, total_docs, width=30)
        line = (
            f"\r{bar} | {progress.chunks_created} chunks | {truncated}"
        )
    else:
        line = (
            f"\rDocs: {progress.documents_processed} | "
            f"Chunks: {progress.chunks_created} | {truncated}"
        )
    print(f"{line:<100}", end="", flush=True)


def print_summary(
    result: IngestProgress,
    elapsed: float,
    indexed: int | None = None,
    verbose: bool = False,
) -> None:
    """Print final summary of ingestion."""
    print()  # newline after progress
    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Documents processed:  {result.documents_processed}")
    print(f"  Chunks created:       {result.chunks_created}")
    if indexed is not None:
        print(f"  Chunks indexed:       {indexed}")
    print(f"  Elapsed time:         {elapsed:.1f}s")
    if result.documents_processed > 0:
        print(
            f"  Throughput:           "
            f"{result.documents_processed / elapsed:.1f} docs/s"
        )

    if result.errors:
        print(f"\n  Errors encountered:   {len(result.errors)}")
        if verbose:
            for error in result.errors:
                print(f"    - {error}")
    else:
        print("\n  No errors encountered.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion pipeline.

    Args:
        argv: Command-line arguments (``None`` for ``sys.argv``).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration
    logger.info("Loading configuration from: %s", args.config)
    settings = Settings(config_path=args.config)

    # Determine corpus path
    corpus_path = args.corpus or settings.corpus.path
    logger.info("Using corpus path: %s", corpus_path)

    if not Path(corpus_path).exists():
        print(f"Error: Corpus path does not exist: {corpus_path}", file=sys.stderr)
        return 1

    try:
        # Create connector
        connector = ConnectorFactory.get_connector(settings.corpus)
        # Override the source path if --corpus was provided
        if args.corpus:
            from src.connectors.local import LocalConnector

            connector = LocalConnector(source_path=args.corpus)

        if not connector.validate_source():
            print(f"Error: Invalid source: {corpus_path}", file=sys.stderr)
            return 1

        # Determine total documents for progress bar
        total_docs: int | None = None
        try:
            total_docs = len(connector.list_documents())
        except Exception:
            pass

        # Create cleaner
        cleaner = TextCleaner()

        # Create chunker
        if args.chunker == "sentence":
            chunker = SentenceChunker(
                min_size=args.min_size,
                max_size=args.max_size,
            )
            chunker_desc = f"sentence (min={args.min_size}, max={args.max_size})"
        else:
            chunker = FixedSizeChunker(
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
            chunker_desc = f"fixed (size={args.chunk_size}, overlap={args.chunk_overlap})"

        # Create pipeline
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=cleaner,
            chunker=chunker,
        )

        print(f"Starting ingestion from: {corpus_path}")
        print(f"Chunker: {chunker_desc}")
        if total_docs:
            print(f"Documents found: {total_docs}")
        print("-" * 60)

        t0 = time.perf_counter()

        if args.no_progress:
            result = pipeline.run()
            final_progress = IngestProgress(
                documents_processed=result.documents_count,
                chunks_created=result.chunks_count,
                errors=result.errors,
            )
        else:
            final_progress = IngestProgress()
            for progress in pipeline.run_with_progress():
                print_progress(progress, total_docs)
                final_progress = progress

        elapsed = time.perf_counter() - t0

        # Optional indexing
        indexed: int | None = None
        if args.index:
            try:
                from src.embeddings.factory import EmbeddingsFactory
                from src.retrieval.service import RetrievalService
                from src.store.factory import VectorStoreFactory

                embeddings = EmbeddingsFactory.get_embeddings(settings.embeddings)
                store = VectorStoreFactory.get_store(
                    settings.vectorstore, dimension=embeddings.dimension,
                )
                retrieval = RetrievalService(embeddings=embeddings, store=store)

                # Collect chunks from a fresh pipeline run (or cache)
                chunks = pipeline.run().chunks if hasattr(pipeline.run(), "chunks") else []
                if chunks:
                    indexed = retrieval.index_documents(chunks)
            except Exception as exc:
                logger.warning("Indexing failed: %s", exc)
                final_progress.errors.append(f"Indexing: {exc}")

        print_summary(final_progress, elapsed, indexed=indexed, verbose=args.verbose)

        if final_progress.errors:
            return 1 if final_progress.documents_processed == 0 else 0
        return 0

    except Exception as exc:
        logger.exception("Ingestion failed with error")
        print(f"\nError: Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
