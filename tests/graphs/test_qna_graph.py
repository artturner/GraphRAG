"""Tests for the Q&A LangGraph workflow assembly."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import GraphConfig
from src.graphs import create_qna_graph
from src.graphs.nodes.answer import BaseLLM
from src.graphs.state import GraphState
from src.store.base import SearchResult
from src.types import Chunk, Citation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str = "c-1", content: str = "Some text") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={"source": "doc.txt"},
    )


def _search_result(
    chunk_id: str = "c-1",
    score: float = 0.9,
    content: str = "Some text",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=_chunk(chunk_id, content),
        metadata={"source": "doc.txt"},
    )


def _citation(chunk_id: str = "c-1", score: float = 0.9) -> Citation:
    return Citation(
        source="doc.txt",
        chunk_id=chunk_id,
        text="Cited text",
        score=score,
    )


class MockLLM:
    """Mock LLM that returns a canned response."""

    def __init__(self, response: str = "Generated answer."):
        self._response = response

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._response


class FailingLLM:
    """Mock LLM that always raises."""

    @property
    def model_name(self) -> str:
        return "failing-llm"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("LLM is unavailable")


def _mock_retrieval(
    results: list[SearchResult] | None = None,
    citations: list[Citation] | None = None,
) -> MagicMock:
    """Create a mock RetrievalService returning the given results."""
    svc = MagicMock()
    results = results if results is not None else []
    citations = citations if citations is not None else []
    svc.search_with_citations.return_value = (results, citations)
    return svc


# ---------------------------------------------------------------------------
# Graph creation
# ---------------------------------------------------------------------------

class TestGraphCreation:
    """Tests that the graph compiles without errors."""

    def test_creates_with_defaults(self):
        graph = create_qna_graph(_mock_retrieval(), MockLLM())
        assert graph is not None

    def test_creates_with_explicit_config(self):
        config = GraphConfig(max_retries=5, refusal_threshold=0.6)
        graph = create_qna_graph(_mock_retrieval(), MockLLM(), config)
        assert graph is not None

    def test_creates_with_zero_retries(self):
        config = GraphConfig(max_retries=0, refusal_threshold=0.5)
        graph = create_qna_graph(_mock_retrieval(), MockLLM(), config)
        assert graph is not None

    def test_graph_is_invocable(self):
        graph = create_qna_graph(_mock_retrieval(), MockLLM())
        assert callable(getattr(graph, "invoke", None))

    def test_graph_has_expected_nodes(self):
        graph = create_qna_graph(_mock_retrieval(), MockLLM())
        node_names = set(graph.get_graph().nodes.keys())
        for name in ("route", "retrieve", "answer", "verify", "retry", "refuse"):
            assert name in node_names


# ---------------------------------------------------------------------------
# Successful path: factual question → retrieve → answer → verify → accept
# ---------------------------------------------------------------------------

class TestSuccessfulPath:
    """Tests the happy path through the graph."""

    def test_factual_question_returns_answer(self):
        sr = _search_result("c-1", 0.9, "Federalism divides power.")
        cit = _citation("c-1", 0.9)
        retrieval = _mock_retrieval(results=[sr], citations=[cit])
        llm = MockLLM("Federalism divides power between governments.")

        # High refusal_threshold requires high confidence — 0.9 > 0.8
        config = GraphConfig(max_retries=2, refusal_threshold=0.8)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is federalism?"})

        assert result["answer"] == "Federalism divides power between governments."
        assert result["query_type"] in ("factual", "procedural")

    def test_procedural_question_returns_answer(self):
        sr = _search_result("c-1", 0.95, "Step 1: install Python.")
        cit = _citation("c-1", 0.95)
        retrieval = _mock_retrieval(results=[sr], citations=[cit])
        llm = MockLLM("First, install Python from python.org.")

        config = GraphConfig(max_retries=2, refusal_threshold=0.8)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "How do I install Python?"})

        assert result["answer"] is not None
        assert result["query_type"] == "procedural"

    def test_high_confidence_accepted(self):
        sr = _search_result("c-1", 0.95, "Content matching answer.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation()])
        llm = MockLLM("Content matching answer.")

        config = GraphConfig(max_retries=2, refusal_threshold=0.5)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is this about?"})

        assert result["answer"] is not None
        assert result["action"] == "accept"

    def test_multiple_search_results(self):
        results = [
            _search_result("c-1", 0.9, "Federalism divides power between governments."),
            _search_result("c-2", 0.85, "Democracy allows citizens to vote."),
        ]
        citations = [_citation("c-1"), _citation("c-2")]
        retrieval = _mock_retrieval(results=results, citations=citations)
        # Answer reuses chunk language so grounding confidence is high
        llm = MockLLM("Federalism divides power between governments.")

        config = GraphConfig(max_retries=2, refusal_threshold=0.5)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What are the passages about?"})

        assert result["answer"] is not None


# ---------------------------------------------------------------------------
# Refusal path: unsupported query → refuse
# ---------------------------------------------------------------------------

class TestRefusalPath:
    """Tests queries that should be refused."""

    def test_unsupported_query_refused(self):
        retrieval = _mock_retrieval()
        llm = MockLLM("Should not be called")
        graph = create_qna_graph(retrieval, llm)

        result = graph.invoke({"question": "Hello there!"})

        assert result["query_type"] == "unsupported"
        assert result["answer"] is None
        assert result.get("refusal_reason") is not None
        # Retrieval should NOT have been called
        retrieval.search_with_citations.assert_not_called()

    def test_greeting_refused(self):
        retrieval = _mock_retrieval()
        llm = MockLLM()
        graph = create_qna_graph(retrieval, llm)

        result = graph.invoke({"question": "Hi"})

        assert result["answer"] is None
        assert result.get("refusal_reason") is not None

    def test_refusal_has_reason(self):
        retrieval = _mock_retrieval()
        llm = MockLLM()
        graph = create_qna_graph(retrieval, llm)

        result = graph.invoke({"question": "ok"})

        reason = result.get("refusal_reason", "")
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# Retry path: low confidence → retry → eventually refuse or accept
# ---------------------------------------------------------------------------

class TestRetryPath:
    """Tests the retry loop."""

    def test_low_confidence_triggers_retry(self):
        """Ungrounded answer → low grounding confidence → retries then refuses."""
        sr = _search_result("c-1", 0.9, "Photosynthesis converts sunlight.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation("c-1", 0.9)])
        # Answer is completely unrelated to chunk → grounding confidence ≈ 0
        llm = MockLLM("Quantum entanglement connects distant particles instantly.")

        config = GraphConfig(max_retries=2, refusal_threshold=0.8)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is something?"})

        # Should have retried and then refused
        assert result["retry_count"] >= 2
        assert result["answer"] is None
        assert result.get("refusal_reason") is not None

    def test_max_retries_zero_goes_straight_to_refuse(self):
        sr = _search_result("c-1", 0.9, "Photosynthesis converts sunlight.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation()])
        # Unrelated answer → low grounding → refuse immediately with max_retries=0
        llm = MockLLM("Quantum computing revolutionizes cryptography.")

        config = GraphConfig(max_retries=0, refusal_threshold=0.8)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is this?"})

        assert result["answer"] is None
        assert result.get("refusal_reason") is not None

    def test_retry_count_increments(self):
        sr = _search_result("c-1", 0.9, "Photosynthesis converts sunlight.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation("c-1", 0.9)])
        # Completely unrelated answer → grounding stays low across retries
        llm = MockLLM("Blockchain decentralizes financial transactions globally.")

        config = GraphConfig(max_retries=3, refusal_threshold=0.9)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is X?"})

        # retry_count should reflect multiple iterations
        assert result["retry_count"] >= 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling within the graph."""

    def test_retrieval_failure_leads_to_refusal(self):
        retrieval = MagicMock()
        retrieval.search_with_citations.side_effect = RuntimeError("DB down")
        llm = MockLLM()

        config = GraphConfig(max_retries=1, refusal_threshold=0.5)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is federalism?"})

        # The error should cause the retry node to refuse
        assert result["answer"] is None

    def test_llm_failure_leads_to_refusal(self):
        sr = _search_result("c-1", 0.9, "Good content.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation()])
        llm = FailingLLM()

        config = GraphConfig(max_retries=1, refusal_threshold=0.5)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is something?"})

        assert result["answer"] is None

    def test_empty_retrieval_results(self):
        retrieval = _mock_retrieval(results=[], citations=[])
        llm = MockLLM("Sorry, I cannot answer.")

        config = GraphConfig(max_retries=1, refusal_threshold=0.5)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is obscure topic?"})

        # Empty results → answer_node produces refusal → verify/retry → refuse
        assert result["answer"] is None


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests that GraphConfig parameters are respected."""

    def test_custom_refusal_threshold_low(self):
        """A very low threshold lets a low-score answer through."""
        sr = _search_result("c-1", 0.3, "Matching content.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation("c-1", 0.3)])
        llm = MockLLM("Matching content.")

        config = GraphConfig(max_retries=2, refusal_threshold=0.1)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is this?"})

        # 0.3 >= 0.1 so it should be accepted
        assert result["answer"] is not None
        assert result["action"] == "accept"

    def test_custom_refusal_threshold_high(self):
        """A very high threshold forces refusal when grounding is imperfect."""
        sr = _search_result("c-1", 0.7, "Photosynthesis converts sunlight.")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation("c-1", 0.7)])
        # Unrelated answer → grounding confidence ≈ 0, way below 0.99
        llm = MockLLM("Quantum computing enables faster calculations.")

        config = GraphConfig(max_retries=1, refusal_threshold=0.99)
        graph = create_qna_graph(retrieval, llm, config)

        result = graph.invoke({"question": "What is this?"})

        assert result["answer"] is None
        assert result.get("refusal_reason") is not None


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify create_qna_graph is importable from the graphs package."""

    def test_create_qna_graph_exported(self):
        from src.graphs import create_qna_graph as fn
        assert callable(fn)

    def test_base_llm_importable(self):
        from src.graphs.nodes.answer import BaseLLM
        assert BaseLLM is not None

    def test_mock_llm_satisfies_protocol(self):
        assert isinstance(MockLLM(), BaseLLM)
