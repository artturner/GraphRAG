#!/usr/bin/env python3
"""
Interactive query CLI for the Grounded GraphRAG Tutor service.

This script lets you pose questions to the RAG pipeline interactively
or in single-shot mode.  An optional ``--debug`` flag prints every
intermediate graph state so you can inspect routing, retrieval,
grounding, and retry decisions.

Usage:
    python scripts/query.py --question "What is federalism?"
    python scripts/query.py --interactive
    python scripts/query.py --question "What is federalism?" --debug
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the query script."""
    parser = argparse.ArgumentParser(
        description="Query the GraphRAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/query.py --question "What is federalism?"
    python scripts/query.py --interactive
    python scripts/query.py --question "What is federalism?" --debug
    python scripts/query.py --question "What is federalism?" --format json
        """,
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="A single question to ask",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive REPL mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate graph state for inspection",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


_DEBUG_FIELDS = [
    "query_type",
    "chunks",
    "search_results",
    "confidence",
    "is_grounded",
    "retry_count",
    "action",
    "refusal_reason",
    "error",
]


def print_debug(state: dict[str, Any]) -> None:
    """Print selected intermediate state fields."""
    print()
    print("----- debug: intermediate state -----")
    for key in _DEBUG_FIELDS:
        val = state.get(key)
        if val is None:
            continue
        # Truncate long lists
        if isinstance(val, list) and len(val) > 3:
            display = f"[{len(val)} items] {str(val[:2])[:120]}..."
        else:
            display = str(val)[:200]
        print(f"  {key}: {display}")
    print("-------------------------------------")
    print()


def print_answer_text(state: dict[str, Any], elapsed_ms: float) -> None:
    """Print the answer in human-readable text format."""
    answer = state.get("answer")
    refusal = state.get("refusal_reason")
    confidence = state.get("confidence", 0.0)
    citations = state.get("citations") or []

    if refusal:
        print(f"\n[REFUSED] {refusal}")
    elif answer:
        print(f"\n{answer}")
    else:
        print("\n[No answer produced]")

    if citations:
        print(f"\nCitations ({len(citations)}):")
        for i, c in enumerate(citations, 1):
            source = getattr(c, "source", str(c))
            print(f"  [{i}] {source}")

    print(f"\n(confidence={confidence:.2f}, latency={elapsed_ms:.0f} ms)")


def print_answer_json(
    question: str, state: dict[str, Any], elapsed_ms: float,
) -> None:
    """Print the answer as a JSON object."""
    citations = state.get("citations") or []
    cit_dicts = []
    for c in citations:
        if hasattr(c, "source"):
            cit_dicts.append({
                "source": c.source,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "score": c.score,
            })
        else:
            cit_dicts.append(str(c))

    obj = {
        "question": question,
        "answer": state.get("answer"),
        "refusal_reason": state.get("refusal_reason"),
        "confidence": state.get("confidence", 0.0),
        "citations": cit_dicts,
        "latency_ms": round(elapsed_ms, 2),
    }
    print(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def run_question(
    question: str,
    graph: Any,
    *,
    debug: bool = False,
    fmt: str = "text",
) -> dict[str, Any]:
    """Invoke the graph with a question and display the result.

    Returns:
        The completed graph state dict.
    """
    t0 = time.perf_counter()
    try:
        state = graph.invoke({"question": question})
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nError: {exc}  (latency={elapsed:.0f} ms)", file=sys.stderr)
        return {"error": str(exc)}
    elapsed = (time.perf_counter() - t0) * 1000

    if debug:
        print_debug(state)

    if fmt == "json":
        print_answer_json(question, state, elapsed)
    else:
        print_answer_text(state, elapsed)

    return state


def interactive_loop(graph: Any, *, debug: bool, fmt: str) -> None:
    """Run an interactive REPL loop."""
    print("GraphRAG Interactive Query  (type 'exit' or Ctrl-C to quit)")
    print()

    while True:
        try:
            question = input("question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        run_question(question, graph, debug=debug, fmt=fmt)
        print()


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------


def build_graph(config_path: str) -> Any:
    """Initialise all services and return a compiled Q&A graph."""
    from src.config import Settings
    from src.embeddings.factory import EmbeddingsFactory
    from src.graphs import create_qna_graph
    from src.llm import LLMFactory
    from src.retrieval.service import RetrievalService
    from src.store.factory import VectorStoreFactory

    settings = Settings(config_path=config_path)

    llm = LLMFactory.get_llm(settings.llm)
    embeddings = EmbeddingsFactory.get_embeddings(settings.embeddings)
    store = VectorStoreFactory.get_store(
        settings.vectorstore, dimension=embeddings.dimension,
    )
    retrieval = RetrievalService(embeddings=embeddings, store=store)
    return create_qna_graph(retrieval, llm, settings.graph)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the query CLI.

    Args:
        argv: Command-line arguments (``None`` for ``sys.argv``).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.question and not args.interactive:
        parser.print_help()
        print(
            "\nError: provide --question or --interactive",
            file=sys.stderr,
        )
        return 1

    # Build pipeline
    try:
        graph = build_graph(args.config)
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        print(
            f"Error: could not initialise the RAG pipeline — {exc}",
            file=sys.stderr,
        )
        return 1

    # Execute
    if args.interactive:
        interactive_loop(graph, debug=args.debug, fmt=args.format)
    else:
        run_question(args.question, graph, debug=args.debug, fmt=args.format)

    return 0


if __name__ == "__main__":
    sys.exit(main())
