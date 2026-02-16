"""Tests for the GraphState definition, StateBuilder, and StateValidator."""

import pytest

from src.graphs import GraphState, StateBuilder, StateValidator
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
    )


def _search_result(chunk_id: str = "c-1", score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=_chunk(chunk_id),
        metadata={},
    )


def _citation(chunk_id: str = "c-1", score: float = 0.9) -> Citation:
    return Citation(
        source="doc.txt",
        chunk_id=chunk_id,
        text="Cited passage",
        score=score,
    )


# ---------------------------------------------------------------------------
# GraphState creation
# ---------------------------------------------------------------------------

class TestGraphStateCreation:
    """Test raw GraphState TypedDict creation."""

    def test_minimal_state(self):
        state: GraphState = {"question": "What is federalism?"}
        assert state["question"] == "What is federalism?"

    def test_full_state(self):
        state: GraphState = {
            "question": "What is federalism?",
            "chunks": [_chunk()],
            "search_results": [_search_result()],
            "answer": "Federalism is...",
            "citations": [_citation()],
            "confidence": 0.92,
            "retry_count": 0,
            "refusal_reason": None,
            "error": None,
        }
        assert state["question"] == "What is federalism?"
        assert state["answer"] == "Federalism is..."
        assert state["confidence"] == 0.92
        assert len(state["chunks"]) == 1
        assert len(state["search_results"]) == 1
        assert len(state["citations"]) == 1

    def test_state_is_mutable_dict(self):
        state: GraphState = {"question": "Q"}
        state["answer"] = "A"
        assert state["answer"] == "A"

    def test_state_supports_update(self):
        state: GraphState = {"question": "Q", "retry_count": 0}
        state["retry_count"] = 1
        assert state["retry_count"] == 1


# ---------------------------------------------------------------------------
# StateBuilder
# ---------------------------------------------------------------------------

class TestStateBuilder:
    """Tests for the fluent StateBuilder helper."""

    def test_build_minimal(self):
        state = StateBuilder().with_question("What is X?").build()
        assert state["question"] == "What is X?"
        assert state["chunks"] == []
        assert state["search_results"] == []
        assert state["answer"] is None
        assert state["citations"] == []
        assert state["confidence"] == 0.0
        assert state["retry_count"] == 0
        assert state["refusal_reason"] is None
        assert state["error"] is None

    def test_build_without_question_raises(self):
        with pytest.raises(ValueError, match="question is required"):
            StateBuilder().build()

    def test_with_question(self):
        state = StateBuilder().with_question("Q?").build()
        assert state["question"] == "Q?"

    def test_with_chunks(self):
        chunks = [_chunk("c-1"), _chunk("c-2")]
        state = StateBuilder().with_question("Q").with_chunks(chunks).build()
        assert len(state["chunks"]) == 2

    def test_with_search_results(self):
        results = [_search_result("c-1", 0.9)]
        state = StateBuilder().with_question("Q").with_search_results(results).build()
        assert len(state["search_results"]) == 1

    def test_with_answer(self):
        state = StateBuilder().with_question("Q").with_answer("A").build()
        assert state["answer"] == "A"

    def test_with_citations(self):
        cits = [_citation()]
        state = StateBuilder().with_question("Q").with_citations(cits).build()
        assert len(state["citations"]) == 1

    def test_with_confidence(self):
        state = StateBuilder().with_question("Q").with_confidence(0.85).build()
        assert state["confidence"] == 0.85

    def test_with_retry_count(self):
        state = StateBuilder().with_question("Q").with_retry_count(2).build()
        assert state["retry_count"] == 2

    def test_with_refusal_reason(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_refusal_reason("Insufficient evidence")
            .build()
        )
        assert state["refusal_reason"] == "Insufficient evidence"

    def test_with_error(self):
        state = StateBuilder().with_question("Q").with_error("LLM timeout").build()
        assert state["error"] == "LLM timeout"

    def test_fluent_chaining(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("A")
            .with_confidence(0.9)
            .with_retry_count(1)
            .build()
        )
        assert state["question"] == "Q"
        assert state["answer"] == "A"
        assert state["confidence"] == 0.9
        assert state["retry_count"] == 1

    def test_builder_produces_independent_states(self):
        builder = StateBuilder().with_question("Q")
        s1 = builder.build()
        s2 = builder.build()
        s1["answer"] = "modified"
        assert s2["answer"] is None


# ---------------------------------------------------------------------------
# StateValidator — validate_initial
# ---------------------------------------------------------------------------

class TestValidateInitial:
    """Tests for StateValidator.validate_initial."""

    def test_valid_initial_state(self):
        state = StateBuilder().with_question("What is X?").build()
        errors = StateValidator().validate_initial(state)
        assert errors == []

    def test_missing_question(self):
        state: GraphState = {}  # type: ignore[typeddict-item]
        errors = StateValidator().validate_initial(state)
        assert any("question" in e for e in errors)

    def test_empty_question(self):
        state: GraphState = {"question": ""}
        errors = StateValidator().validate_initial(state)
        assert any("question" in e for e in errors)

    def test_whitespace_question(self):
        state: GraphState = {"question": "   "}
        errors = StateValidator().validate_initial(state)
        assert any("question" in e for e in errors)


# ---------------------------------------------------------------------------
# StateValidator — validate_for_retrieval
# ---------------------------------------------------------------------------

class TestValidateForRetrieval:
    """Tests for StateValidator.validate_for_retrieval."""

    def test_valid(self):
        state = StateBuilder().with_question("Q").build()
        assert StateValidator().validate_for_retrieval(state) == []

    def test_inherits_initial_checks(self):
        state: GraphState = {"question": ""}
        errors = StateValidator().validate_for_retrieval(state)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# StateValidator — validate_for_answer
# ---------------------------------------------------------------------------

class TestValidateForAnswer:
    """Tests for StateValidator.validate_for_answer."""

    def test_valid_with_results(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([_search_result()])
            .build()
        )
        assert StateValidator().validate_for_answer(state) == []

    def test_valid_with_empty_results(self):
        state = StateBuilder().with_question("Q").build()
        # search_results is present (empty list) via builder
        assert StateValidator().validate_for_answer(state) == []

    def test_missing_search_results_key(self):
        state: GraphState = {"question": "Q"}
        errors = StateValidator().validate_for_answer(state)
        assert any("search_results" in e for e in errors)


# ---------------------------------------------------------------------------
# StateValidator — validate_for_grounding
# ---------------------------------------------------------------------------

class TestValidateForGrounding:
    """Tests for StateValidator.validate_for_grounding."""

    def test_valid(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([])
            .with_answer("Some answer")
            .build()
        )
        assert StateValidator().validate_for_grounding(state) == []

    def test_missing_answer(self):
        state = StateBuilder().with_question("Q").build()
        errors = StateValidator().validate_for_grounding(state)
        assert any("answer" in e for e in errors)

    def test_empty_answer(self):
        state = StateBuilder().with_question("Q").with_answer("").build()
        # answer is "" which is falsy
        errors = StateValidator().validate_for_grounding(state)
        assert any("answer" in e for e in errors)

    def test_whitespace_answer(self):
        state: GraphState = {
            "question": "Q",
            "search_results": [],
            "answer": "   ",
        }
        errors = StateValidator().validate_for_grounding(state)
        assert any("answer" in e for e in errors)


# ---------------------------------------------------------------------------
# StateValidator — validate_complete
# ---------------------------------------------------------------------------

class TestValidateComplete:
    """Tests for StateValidator.validate_complete."""

    def test_complete_with_answer(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("A")
            .with_confidence(0.9)
            .build()
        )
        assert StateValidator().validate_complete(state) == []

    def test_complete_with_refusal(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_refusal_reason("Not enough evidence")
            .build()
        )
        assert StateValidator().validate_complete(state) == []

    def test_complete_with_error(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_error("LLM unavailable")
            .build()
        )
        assert StateValidator().validate_complete(state) == []

    def test_incomplete_no_outcome(self):
        state = StateBuilder().with_question("Q").build()
        errors = StateValidator().validate_complete(state)
        assert any("answer" in e or "refusal" in e or "error" in e for e in errors)

    def test_answer_with_invalid_confidence(self):
        state: GraphState = {
            "question": "Q",
            "answer": "A",
            "confidence": 1.5,
        }
        errors = StateValidator().validate_complete(state)
        assert any("confidence" in e for e in errors)

    def test_answer_with_negative_confidence(self):
        state: GraphState = {
            "question": "Q",
            "answer": "A",
            "confidence": -0.1,
        }
        errors = StateValidator().validate_complete(state)
        assert any("confidence" in e for e in errors)

    def test_answer_with_zero_confidence_is_valid(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("A")
            .with_confidence(0.0)
            .build()
        )
        # confidence=0.0 is in [0,1] so no confidence error
        assert StateValidator().validate_complete(state) == []


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify all graph state classes are importable from the package."""

    def test_graph_state_exported(self):
        from src.graphs import GraphState
        assert GraphState is not None

    def test_state_builder_exported(self):
        from src.graphs import StateBuilder
        assert StateBuilder is not None

    def test_state_validator_exported(self):
        from src.graphs import StateValidator
        assert StateValidator is not None
