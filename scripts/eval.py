#!/usr/bin/env python3
"""
Evaluation script for the Grounded GraphRAG Tutor service.

This script runs evaluation datasets through the RAG pipeline and measures
quality metrics including groundedness, relevance, and refusal correctness.

Usage:
    python scripts/eval.py --suite sample_qna --output reports/
    python scripts/eval.py --dataset path/to/custom.yaml --output reports/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval import EvalRunner, load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in suite lookup
# ---------------------------------------------------------------------------

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "src" / "eval" / "datasets"

_SUITES: dict[str, Path] = {
    "sample_qna": _DATASETS_DIR / "sample_qna.yaml",
}


def _resolve_dataset_path(args: argparse.Namespace) -> Path:
    """Determine the dataset file path from CLI arguments."""
    if args.dataset:
        p = Path(args.dataset)
        if not p.exists():
            print(f"Error: dataset file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p

    suite = args.suite or "sample_qna"
    if suite not in _SUITES:
        print(
            f"Error: unknown suite {suite!r}. "
            f"Available: {', '.join(sorted(_SUITES))}",
            file=sys.stderr,
        )
        sys.exit(1)
    return _SUITES[suite]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Evaluate the GraphRAG system quality",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Name of a built-in evaluation suite (e.g. sample_qna)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a custom evaluation dataset (YAML or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to write report files (JSON + HTML)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve dataset
    dataset_path = _resolve_dataset_path(args)
    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset(dataset_path)

    # Build the graph — lazy-import to avoid heavy dependencies when
    # only running --help or validation.
    try:
        from src.config import settings
        from src.graphs import create_qna_graph
        from src.llm import LLMFactory
        from src.retrieval.service import RetrievalService

        llm = LLMFactory.create(settings.llm.provider, settings.llm.model_name)
        retrieval = RetrievalService(settings)
        graph = create_qna_graph(retrieval, llm, settings.graph)
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        print(f"Error: could not initialise the RAG pipeline — {exc}", file=sys.stderr)
        return 1

    # Run evaluation
    runner = EvalRunner()
    report = runner.run(dataset, graph)

    # Print summary
    m = report.metrics
    print()
    print(f"=== Evaluation Report: {report.suite_name} ===")
    print(f"  Total questions:    {m.total_questions}")
    print(f"  Answered:           {m.answered_count}")
    print(f"  Refused:            {m.refused_count}")
    print(f"  Errors:             {m.error_count}")
    print(f"  Avg groundedness:   {m.avg_groundedness:.4f}")
    print(f"  Avg relevance:      {m.avg_relevance:.4f}")
    print(f"  Refusal accuracy:   {m.refusal_accuracy:.2%}")
    print(f"  Avg latency:        {m.avg_latency_ms:.0f} ms")
    print()

    # Save report
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{report.suite_name}.json"
        json_path.write_text(report.to_json(), encoding="utf-8")
        print(f"JSON report saved to {json_path}")

        html_path = out_dir / f"{report.suite_name}.html"
        html_path.write_text(report.to_html(), encoding="utf-8")
        print(f"HTML report saved to {html_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
