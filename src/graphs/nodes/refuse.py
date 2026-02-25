"""Refusal node for the LangGraph Q&A workflow.

This node produces a polite, informative refusal when the workflow
cannot generate a satisfactory answer.
"""

import logging
import re

from src.graphs.state import GraphState

_STRUCTURAL_RE = re.compile(
    r"\bchapter\s+\d+\b|\bch\.?\s*\d+\b|\bpage\s+\d+\b|\bp\.\s*\d+\b|\bsection\s+\d+[\.\d]*\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


def _build_refusal_reason(state: GraphState) -> str:
    """Synthesise a human-readable refusal reason from state signals."""
    question = state.get("question", "your question")
    error = state.get("error")
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    chunks = state.get("chunks", [])
    search_results = state.get("search_results", [])

    parts: list[str] = []

    if _STRUCTURAL_RE.search(question):
        parts.append(
            "This system uses semantic search and cannot look up content by "
            "chapter, page, or section number. "
            "Try asking about a specific topic instead — for example, "
            "\"Summarize the key points of bicameralism\" rather than "
            "\"Give me the key points of chapter 2\"."
        )
        return " ".join(parts)

    if not search_results and not chunks:
        parts.append(
            "No relevant documents were found in the knowledge base "
            f"for: \"{question}\""
        )
    elif confidence > 0:
        parts.append(
            f"The available evidence was not strong enough to provide "
            f"a reliable answer (confidence: {confidence:.0%})."
        )
    else:
        parts.append(
            "The retrieved documents did not contain sufficient "
            f"information to answer: \"{question}\""
        )

    if error:
        parts.append(f"Additionally, an error occurred: {error}")

    if retry_count > 1:
        parts.append(
            f"The system attempted {retry_count} times before giving up."
        )

    parts.append(
        "Please try rephrasing your question or providing more context."
    )

    return " ".join(parts)


def refuse_node(state: GraphState) -> dict:
    """Generate a refusal response.

    This node is reached when the workflow has determined it cannot
    produce a trustworthy answer — either because no relevant context
    was found, the confidence was too low after retries, or an
    unrecoverable error occurred.

    Args:
        state: Current graph state.

    Returns:
        A dict setting ``answer`` to ``None``, ``confidence`` to 0,
        and a descriptive ``refusal_reason``.
    """
    reason = _build_refusal_reason(state)

    logger.info("Refusing to answer: %s", reason[:120])

    return {
        "answer": None,
        "confidence": 0.0,
        "refusal_reason": reason,
    }
