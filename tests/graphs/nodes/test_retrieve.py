"""Tests for the retrieve node."""

from unittest.mock import MagicMock

import pytest

from src.exceptions import RetrievalError
from src.graphs.nodes import retrieve_node
from src.graphs.state import GraphState, StateBuilder
from src.store.base import SearchResult
from src.types import Chunk, Citation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str = "c-1", content: str = "Some text", source: str = "doc.txt") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-001",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={"source": source},
    )


def _search_result(
    chunk_id: str = "c-1",
    score: float = 0.9,
    content: str = "Some text",
    source: str = "doc.txt",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=_chunk(chunk_id, content, source),
        metadata={"source": source},
    )


def _citation(chunk_id: str = "c-1", score: float = 0.9) -> Citation:
    return Citation(
        source="doc.txt",
        chunk_id=chunk_id,
        text="Cited text",
        score=score,
    )


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
# Basic retrieval
# ---------------------------------------------------------------------------

class TestRetrieveNodeBasic:
    """Tests for successful retrieval."""

    def test_returns_chunks_and_results(self):
        sr = _search_result("c-1", 0.95, "Federalism is a system.")
        cit = _citation("c-1", 0.95)
        retrieval = _mock_retrieval(results=[sr], citations=[cit])
        state: GraphState = {"question": "What is federalism?"}

        out = retrieve_node(state, retrieval)

        assert len(out["chunks"]) == 1
        assert len(out["search_results"]) == 1
        assert len(out["citations"]) == 1
        assert "error" not in out

    def test_chunks_extracted_from_results(self):
        sr = _search_result("c-1", 0.9, "Content here")
        retrieval = _mock_retrieval(results=[sr], citations=[_citation()])
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert out["chunks"][0].id == "c-1"
        assert out["chunks"][0].content == "Content here"

    def test_multiple_results(self):
        results = [
            _search_result("c-1", 0.95, "First"),
            _search_result("c-2", 0.85, "Second"),
            _search_result("c-3", 0.70, "Third"),
        ]
        citations = [_citation("c-1"), _citation("c-2"), _citation("c-3")]
        retrieval = _mock_retrieval(results=results, citations=citations)
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert len(out["chunks"]) == 3
        assert len(out["search_results"]) == 3
        assert len(out["citations"]) == 3

    def test_passes_question_to_service(self):
        retrieval = _mock_retrieval()
        state: GraphState = {"question": "What is democracy?"}

        retrieve_node(state, retrieval)

        retrieval.search_with_citations.assert_called_once_with(
            "What is democracy?", k=5
        )

    def test_custom_k(self):
        retrieval = _mock_retrieval()
        state: GraphState = {"question": "Q?"}

        retrieve_node(state, retrieval, k=10)

        retrieval.search_with_citations.assert_called_once_with("Q?", k=10)

    def test_works_with_state_builder(self):
        retrieval = _mock_retrieval(
            results=[_search_result()], citations=[_citation()]
        )
        state = StateBuilder().with_question("Q?").build()

        out = retrieve_node(state, retrieval)

        assert len(out["chunks"]) == 1


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

class TestRetrieveNodeEmptyResults:
    """Tests for handling of empty result sets."""

    def test_empty_results_from_service(self):
        retrieval = _mock_retrieval(results=[], citations=[])
        state: GraphState = {"question": "Something obscure?"}

        out = retrieve_node(state, retrieval)

        assert out["chunks"] == []
        assert out["search_results"] == []
        assert out["citations"] == []
        assert "error" not in out

    def test_empty_question(self):
        retrieval = _mock_retrieval()
        state: GraphState = {"question": ""}

        out = retrieve_node(state, retrieval)

        assert out["chunks"] == []
        assert out["search_results"] == []
        assert out["citations"] == []
        assert "error" in out
        # Service should NOT be called
        retrieval.search_with_citations.assert_not_called()

    def test_whitespace_question(self):
        retrieval = _mock_retrieval()
        state: GraphState = {"question": "   "}

        out = retrieve_node(state, retrieval)

        assert "error" in out
        retrieval.search_with_citations.assert_not_called()

    def test_missing_question_key(self):
        retrieval = _mock_retrieval()
        state: GraphState = {}  # type: ignore[typeddict-item]

        out = retrieve_node(state, retrieval)

        assert "error" in out
        retrieval.search_with_citations.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRetrieveNodeErrors:
    """Tests for error handling when retrieval fails."""

    def test_retrieval_error_caught(self):
        retrieval = MagicMock()
        retrieval.search_with_citations.side_effect = RetrievalError(
            "Search failed", details="connection timeout"
        )
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert out["chunks"] == []
        assert out["search_results"] == []
        assert out["citations"] == []
        assert "error" in out
        assert "Retrieval failed" in out["error"]

    def test_generic_exception_caught(self):
        retrieval = MagicMock()
        retrieval.search_with_citations.side_effect = RuntimeError("boom")
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert out["chunks"] == []
        assert "error" in out
        assert "boom" in out["error"]

    def test_error_state_has_all_required_keys(self):
        retrieval = MagicMock()
        retrieval.search_with_citations.side_effect = Exception("fail")
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert "chunks" in out
        assert "search_results" in out
        assert "citations" in out
        assert "error" in out


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestRetrieveNodeReturnStructure:
    """Verify the shape of the returned dict."""

    def test_success_keys(self):
        retrieval = _mock_retrieval(results=[_search_result()], citations=[_citation()])
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert set(out.keys()) == {"chunks", "search_results", "citations"}

    def test_error_keys(self):
        retrieval = MagicMock()
        retrieval.search_with_citations.side_effect = Exception("fail")
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert set(out.keys()) == {"chunks", "search_results", "citations", "error"}

    def test_chunks_are_chunk_objects(self):
        retrieval = _mock_retrieval(results=[_search_result()], citations=[_citation()])
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert all(isinstance(c, Chunk) for c in out["chunks"])

    def test_search_results_are_search_result_objects(self):
        retrieval = _mock_retrieval(results=[_search_result()], citations=[_citation()])
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert all(isinstance(r, SearchResult) for r in out["search_results"])

    def test_citations_are_citation_objects(self):
        retrieval = _mock_retrieval(results=[_search_result()], citations=[_citation()])
        state: GraphState = {"question": "Q?"}

        out = retrieve_node(state, retrieval)

        assert all(isinstance(c, Citation) for c in out["citations"])


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify retrieve_node is importable from the nodes package."""

    def test_retrieve_node_exported(self):
        from src.graphs.nodes import retrieve_node as fn
        assert callable(fn)
