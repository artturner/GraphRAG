"""Retrieve node for fetching relevant document chunks.

This node uses the RetrievalService to search for chunks relevant to
the user's question and populates the graph state with the results.
"""

import logging

from src.graphs.state import GraphState
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

# Default number of results to retrieve
DEFAULT_K = 5


def retrieve_node(
    state: GraphState,
    retrieval: RetrievalService,
    k: int = DEFAULT_K,
) -> dict:
    """Retrieve document chunks relevant to the question in *state*.

    The node calls ``retrieval.search`` with the question and returns
    a dict containing ``chunks``, ``search_results``, and ``citations``.
    If the search returns no results, the lists will be empty.  If the
    retrieval service raises an exception the node catches it and sets
    ``error`` instead.

    Args:
        state: Current graph state — must contain ``question``.
        retrieval: The retrieval service to query.
        k: Maximum number of results to retrieve.

    Returns:
        A dict with keys ``chunks``, ``search_results``, and
        ``citations``.  On failure, also includes ``error``.

    Example:
        ```python
        result = retrieve_node(
            {"question": "What is federalism?"},
            retrieval_service,
        )
        # result == {"chunks": [...], "search_results": [...], "citations": [...]}
        ```
    """
    question: str = state.get("question", "")

    if not question or not question.strip():
        logger.warning("retrieve_node received empty question")
        return {
            "chunks": [],
            "search_results": [],
            "citations": [],
            "error": "Cannot retrieve: question is empty",
        }

    try:
        results, citations = retrieval.search_with_citations(question, k=k)

        chunks = [r.chunk for r in results]

        logger.debug(
            "Retrieved %d chunks for question: %s",
            len(chunks),
            question[:80],
        )

        return {
            "chunks": chunks,
            "search_results": results,
            "citations": citations,
        }

    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {
            "chunks": [],
            "search_results": [],
            "citations": [],
            "error": f"Retrieval failed: {e}",
        }
