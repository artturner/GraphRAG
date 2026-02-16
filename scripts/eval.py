#!/usr/bin/env python3
"""
Evaluation script for the Grounded GraphRAG Tutor service.

This script runs evaluation datasets through the RAG pipeline and measures
quality metrics including groundedness, relevance, and refusal correctness.

Usage:
    python scripts/eval.py --suite sample_qna --output reports/
    python scripts/eval.py --dataset path/to/custom.yaml --output reports/
    python scripts/eval.py --suite sample_qna --format json
    python scripts/eval.py --suite sample_qna --baseline reports/baseline.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval import EvalReport, EvalRunner, load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in suite lookup
# ---------------------------------------------------------------------------

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "src" / "eval" / "datasets"

_SUITES: dict[str, Path] = {
    "sample_qna": _DATASETS_DIR / "sample_qna.yaml",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate the GraphRAG system quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval.py --suite sample_qna
    python scripts/eval.py --suite sample_qna --output reports/
    python scripts/eval.py --dataset path/to/custom.yaml --format json
    python scripts/eval.py --suite sample_qna --baseline reports/baseline.json
        """,
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
        help="Directory to write report files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "html", "all"],
        default="all",
        help="Output format when --output is set (default: all)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to a baseline report JSON for comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def resolve_dataset_path(args: argparse.Namespace) -> Path:
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
# Baseline comparison
# ---------------------------------------------------------------------------


def load_baseline(path: str) -> dict | None:
    """Load a baseline report JSON for comparison."""
    p = Path(path)
    if not p.exists():
        print(f"Warning: baseline file not found: {p}", file=sys.stderr)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("metrics", {})
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"Warning: could not parse baseline: {exc}", file=sys.stderr)
        return None


def _delta(current: float, baseline: float) -> str:
    """Format a delta comparison string."""
    diff = current - baseline
    if abs(diff) < 0.0001:
        return " (=)"
    sign = "+" if diff > 0 else ""
    return f" ({sign}{diff:.4f})"


def print_summary(report: EvalReport, baseline: dict | None = None) -> None:
    """Print a formatted summary to stdout."""
    m = report.metrics
    print()
    print(f"=== Evaluation Report: {report.suite_name} ===")
    print(f"  Total questions:    {m.total_questions}")
    print(f"  Answered:           {m.answered_count}")
    print(f"  Refused:            {m.refused_count}")
    print(f"  Errors:             {m.error_count}")

    g_delta = _delta(m.avg_groundedness, baseline["avg_groundedness"]) if baseline else ""
    r_delta = _delta(m.avg_relevance, baseline["avg_relevance"]) if baseline else ""
    ref_delta = _delta(m.refusal_accuracy, baseline["refusal_accuracy"]) if baseline else ""
    lat_delta = ""
    if baseline and "avg_latency_ms" in baseline:
        lat_diff = m.avg_latency_ms - baseline["avg_latency_ms"]
        sign = "+" if lat_diff > 0 else ""
        lat_delta = f" ({sign}{lat_diff:.0f} ms)"

    print(f"  Avg groundedness:   {m.avg_groundedness:.4f}{g_delta}")
    print(f"  Avg relevance:      {m.avg_relevance:.4f}{r_delta}")
    print(f"  Refusal accuracy:   {m.refusal_accuracy:.2%}{ref_delta}")
    print(f"  Avg latency:        {m.avg_latency_ms:.0f} ms{lat_delta}")

    if baseline:
        print()
        print("  (compared against baseline)")
    print()


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------


def save_report(
    report: EvalReport,
    output_dir: str,
    fmt: str,
) -> None:
    """Save the report to the output directory in the requested format(s)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if fmt in ("json", "all"):
        path = out / f"{report.suite_name}.json"
        path.write_text(report.to_json(), encoding="utf-8")
        print(f"JSON report saved to {path}")

    if fmt in ("html", "all"):
        path = out / f"{report.suite_name}.html"
        path.write_text(report.to_html(), encoding="utf-8")
        print(f"HTML report saved to {path}")

    if fmt in ("text", "all"):
        path = out / f"{report.suite_name}.txt"
        lines = [
            f"Evaluation Report: {report.suite_name}",
            f"Generated: {report.created_at}",
            "",
            f"Total questions:  {report.metrics.total_questions}",
            f"Answered:         {report.metrics.answered_count}",
            f"Refused:          {report.metrics.refused_count}",
            f"Errors:           {report.metrics.error_count}",
            f"Avg groundedness: {report.metrics.avg_groundedness:.4f}",
            f"Avg relevance:    {report.metrics.avg_relevance:.4f}",
            f"Refusal accuracy: {report.metrics.refusal_accuracy:.2%}",
            f"Avg latency:      {report.metrics.avg_latency_ms:.0f} ms",
            "",
            "Per-question results:",
        ]
        for r in report.results:
            status = "ERROR" if r.error else ("REFUSED" if r.refusal_reason else "OK")
            g = f"{r.groundedness:.2f}" if r.groundedness is not None else "—"
            rv = f"{r.relevance:.2f}" if r.relevance is not None else "—"
            lines.append(
                f"  [{status:>7}] {r.question[:60]:<60}  G={g}  R={rv}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Text report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation harness.

    Args:
        argv: Command-line arguments (``None`` for ``sys.argv``).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve dataset
    dataset_path = resolve_dataset_path(args)
    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset(dataset_path)

    # Build the graph — lazy-import to avoid heavy dependencies when
    # only running --help or validation.
    try:
        from src.config import settings
        from src.embeddings.factory import EmbeddingsFactory
        from src.graphs import create_qna_graph
        from src.llm import LLMFactory
        from src.retrieval.service import RetrievalService
        from src.store.factory import VectorStoreFactory

        llm = LLMFactory.get_llm(settings.llm)
        embeddings = EmbeddingsFactory.get_embeddings(settings.embeddings)
        store = VectorStoreFactory.get_store(
            settings.vectorstore, dimension=embeddings.dimension,
        )
        retrieval = RetrievalService(
            embeddings=embeddings, store=store,
        )
        graph = create_qna_graph(retrieval, llm, settings.graph)
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        print(
            f"Error: could not initialise the RAG pipeline — {exc}",
            file=sys.stderr,
        )
        return 1

    # Run evaluation
    runner = EvalRunner()
    report = runner.run(dataset, graph)

    # Load baseline for comparison (optional)
    baseline = load_baseline(args.baseline) if args.baseline else None

    # Print summary
    print_summary(report, baseline=baseline)

    # Save report
    if args.output:
        save_report(report, args.output, args.format)

    return 0


if __name__ == "__main__":
    sys.exit(main())
