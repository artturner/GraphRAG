"""Tests for the refuse node."""

import pytest

from src.graphs.nodes import refuse_node
from src.graphs.state import GraphState, StateBuilder
from src.store.base import SearchResult
from src.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content: str = "text") -> Chunk:
    return Chunk(
        id="c-1",
        document_id="doc-1",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={"source": "doc.txt"},
    )


def _search_result(content: str = "text") -> SearchResult:
    return SearchResult(
        chunk_id="c-1",
        score=0.5,
        chunk=_chunk(content),
        metadata={"source": "doc.txt"},
    )


# ---------------------------------------------------------------------------
# Refusal response generation
# ---------------------------------------------------------------------------

class TestRefuseNodeResponse:
    """Tests that refuse_node produces the expected output shape."""

    def test_returns_none_answer(self):
        state = StateBuilder().with_question("Q?").build()
        out = refuse_node(state)
        assert out["answer"] is None

    def test_returns_zero_confidence(self):
        state = StateBuilder().with_question("Q?").build()
        out = refuse_node(state)
        assert out["confidence"] == 0.0

    def test_returns_refusal_reason(self):
        state = StateBuilder().with_question("Q?").build()
        out = refuse_node(state)
        assert "refusal_reason" in out
        assert isinstance(out["refusal_reason"], str)
        assert len(out["refusal_reason"]) > 0

    def test_output_keys(self):
        state = StateBuilder().with_question("Q?").build()
        out = refuse_node(state)
        assert set(out.keys()) == {"answer", "confidence", "refusal_reason"}


# ---------------------------------------------------------------------------
# Refusal reason is helpful
# ---------------------------------------------------------------------------

class TestRefuseNodeReasonQuality:
    """Tests that the refusal reason is informative and helpful."""

    def test_includes_question(self):
        state = StateBuilder().with_question("What is federalism?").build()
        out = refuse_node(state)
        assert "federalism" in out["refusal_reason"].lower()

    def test_suggests_rephrasing(self):
        state = StateBuilder().with_question("Q?").build()
        out = refuse_node(state)
        assert "rephras" in out["refusal_reason"].lower()

    def test_no_documents_message(self):
        state = StateBuilder().with_question("Obscure topic?").build()
        out = refuse_node(state)
        assert "no relevant" in out["refusal_reason"].lower()

    def test_low_confidence_message(self):
        state = (
            StateBuilder()
            .with_question("Q?")
            .with_confidence(0.3)
            .with_search_results([_search_result()])
            .with_chunks([_chunk()])
            .build()
        )
        out = refuse_node(state)
        assert "not strong enough" in out["refusal_reason"].lower() or "confidence" in out["refusal_reason"].lower()

    def test_error_included_in_reason(self):
        state = (
            StateBuilder()
            .with_question("Q?")
            .with_error("LLM service timeout")
            .build()
        )
        out = refuse_node(state)
        assert "LLM service timeout" in out["refusal_reason"]

    def test_retry_count_mentioned(self):
        state = (
            StateBuilder()
            .with_question("Q?")
            .with_retry_count(3)
            .build()
        )
        out = refuse_node(state)
        assert "3" in out["refusal_reason"]

    def test_single_retry_not_mentioned(self):
        """A single attempt (retry_count=1) shouldn't say 'attempted N times'."""
        state = (
            StateBuilder()
            .with_question("Q?")
            .with_retry_count(1)
            .build()
        )
        out = refuse_node(state)
        # retry_count <= 1 should not trigger the "attempted X times" message
        assert "attempted" not in out["refusal_reason"].lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRefuseNodeEdgeCases:
    """Tests for edge-case inputs."""

    def test_minimal_state(self):
        state: GraphState = {"question": "Q?"}
        out = refuse_node(state)
        assert out["answer"] is None
        assert "refusal_reason" in out

    def test_empty_question(self):
        state: GraphState = {"question": ""}
        out = refuse_node(state)
        assert out["answer"] is None
        assert "refusal_reason" in out

    def test_missing_question(self):
        state: GraphState = {}  # type: ignore[typeddict-item]
        out = refuse_node(state)
        assert out["answer"] is None
        assert "refusal_reason" in out

    def test_with_chunks_but_no_search_results(self):
        state = (
            StateBuilder()
            .with_question("Q?")
            .with_chunks([_chunk()])
            .build()
        )
        out = refuse_node(state)
        assert out["answer"] is None
        assert "refusal_reason" in out

    def test_with_existing_answer_ignored(self):
        """Even if state has an answer, refuse_node overrides it."""
        state = (
            StateBuilder()
            .with_question("Q?")
            .with_answer("Old answer")
            .build()
        )
        out = refuse_node(state)
        assert out["answer"] is None

    def test_with_all_signals(self):
        state = (
            StateBuilder()
            .with_question("Complex question?")
            .with_confidence(0.2)
            .with_retry_count(5)
            .with_error("timeout")
            .with_search_results([_search_result()])
            .with_chunks([_chunk()])
            .build()
        )
        out = refuse_node(state)
        assert out["answer"] is None
        assert out["confidence"] == 0.0
        reason = out["refusal_reason"]
        assert "timeout" in reason
        assert "5" in reason


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify refuse_node is importable from the nodes package."""

    def test_refuse_node_exported(self):
        from src.graphs.nodes import refuse_node as fn
        assert callable(fn)
