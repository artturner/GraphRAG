"""LangGraph Q&A workflow assembly.

This module wires every node into a :class:`~langgraph.graph.StateGraph`,
adds conditional edges for the retry/refuse loop, and exposes a single
factory function :func:`create_qna_graph`.

Flow::

    START
      │
      ▼
    route ──unsupported──► refuse ──► END
      │
      │ factual / procedural / synthesis / summarize
      ▼
    retrieve
      │
      ▼
    answer ──synthesis / summarize──► END
      │
      │ factual / procedural
      ▼
    verify
      │
      ▼
    retry ──accept──► END
      │        │
      │      refuse
      │        │
      │        ▼
      │      refuse ──► END
      │
      │ retry (loop back)
      └──────► retrieve
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import GraphConfig
from src.graphs.grounding import GroundingChecker
from src.graphs.nodes.answer import BaseLLM, answer_node
from src.graphs.nodes.refuse import refuse_node
from src.graphs.nodes.retrieve import retrieve_node
from src.graphs.nodes.retry import retry_node
from src.graphs.nodes.route import route_node
from src.graphs.nodes.verify import verify_grounding_node
from src.graphs.state import GraphState
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal routing helpers
# ---------------------------------------------------------------------------

def _route_decision(state: GraphState) -> str:
    """Return the next node name based on query type."""
    query_type = state.get("query_type", "unsupported")
    if query_type == "unsupported":
        return "refuse"
    return "retrieve"


def _answer_decision(state: GraphState) -> str:
    """Skip verify/retry for generative queries — go straight to END."""
    if state.get("query_type") in ("synthesis", "summarize"):
        return END
    return "verify"


def _retry_decision(state: GraphState) -> str:
    """Return the next node name based on the retry action."""
    action = state.get("action", "refuse")
    if action == "accept":
        return END
    if action == "retry":
        return "retrieve"
    # "refuse" or anything unexpected
    return "refuse"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_qna_graph(
    retrieval: RetrievalService,
    llm: BaseLLM,
    config: GraphConfig | None = None,
) -> Any:
    """Build and compile the Q&A LangGraph workflow.

    Args:
        retrieval: The retrieval service used by the retrieve node.
        llm: An object implementing :class:`BaseLLM` for answer generation.
        config: Optional :class:`GraphConfig`.  When ``None`` the
            defaults are used (``max_retries=2``, ``refusal_threshold=0.8``).

    Returns:
        A compiled LangGraph ``CompiledGraph`` that accepts
        ``{"question": "..."}`` and returns a completed ``GraphState``.

    Example:
        ```python
        graph = create_qna_graph(retrieval_service, llm)
        result = graph.invoke({"question": "What is federalism?"})
        print(result["answer"])
        ```
    """
    if config is None:
        config = GraphConfig()

    # -- Bind extra arguments into single-arg callables ----------------
    def _retrieve(state: GraphState) -> dict:
        return retrieve_node(state, retrieval)

    def _answer(state: GraphState) -> dict:
        return answer_node(state, llm)

    def _verify(state: GraphState) -> dict:
        return verify_grounding_node(state)

    def _retry(state: GraphState) -> dict:
        return retry_node(
            state,
            max_retries=config.max_retries,
            min_confidence=config.refusal_threshold,
        )

    # -- Build graph ---------------------------------------------------
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("answer", _answer)
    graph.add_node("verify", _verify)
    graph.add_node("retry", _retry)
    graph.add_node("refuse", refuse_node)

    # -- Edges ---------------------------------------------------------
    graph.add_edge(START, "route")

    graph.add_conditional_edges(
        "route",
        _route_decision,
        {"retrieve": "retrieve", "refuse": "refuse"},
    )

    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges(
        "answer",
        _answer_decision,
        {"verify": "verify", END: END},
    )
    graph.add_edge("verify", "retry")

    graph.add_conditional_edges(
        "retry",
        _retry_decision,
        {END: END, "retrieve": "retrieve", "refuse": "refuse"},
    )

    graph.add_edge("refuse", END)

    # -- Compile -------------------------------------------------------
    compiled = graph.compile()

    logger.info(
        "Q&A graph compiled (max_retries=%d, threshold=%.2f)",
        config.max_retries,
        config.refusal_threshold,
    )

    return compiled
