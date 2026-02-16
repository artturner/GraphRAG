"""Tests for the answer node and prompt utilities."""

from unittest.mock import MagicMock

import pytest

from src.graphs.nodes import answer_node
from src.graphs.nodes.answer import BaseLLM
from src.graphs.prompts import (
    RAG_PROMPT,
    REFUSAL_PROMPT,
    build_rag_prompt,
    build_refusal_prompt,
)
from src.graphs.state import GraphState, StateBuilder
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


class MockLLM:
    """Mock LLM that echoes a canned response."""

    def __init__(self, response: str = "Generated answer."):
        self._response = response
        self.last_prompt: str | None = None
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        return self._response


class FailingLLM:
    """Mock LLM that always raises."""

    @property
    def model_name(self) -> str:
        return "failing-llm"

    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("LLM is unavailable")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class TestRAGPrompt:
    """Tests for the RAG_PROMPT template."""

    def test_contains_placeholders(self):
        assert "{context}" in RAG_PROMPT
        assert "{question}" in RAG_PROMPT

    def test_format_produces_valid_string(self):
        prompt = RAG_PROMPT.format(context="ctx", question="q")
        assert "ctx" in prompt
        assert "q" in prompt


class TestRefusalPrompt:
    """Tests for the REFUSAL_PROMPT template."""

    def test_contains_placeholder(self):
        assert "{question}" in REFUSAL_PROMPT

    def test_format_produces_valid_string(self):
        prompt = REFUSAL_PROMPT.format(question="q")
        assert "q" in prompt


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    """Tests for the build_rag_prompt utility."""

    def test_includes_question(self):
        prompt = build_rag_prompt("What is X?", ["text"])
        assert "What is X?" in prompt

    def test_includes_context(self):
        prompt = build_rag_prompt("Q", ["Alpha", "Beta"])
        assert "Alpha" in prompt
        assert "Beta" in prompt

    def test_numbers_context_passages(self):
        prompt = build_rag_prompt("Q", ["First", "Second"])
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_empty_context(self):
        prompt = build_rag_prompt("Q", [])
        assert "No relevant context" in prompt

    def test_single_context(self):
        prompt = build_rag_prompt("Q", ["Only passage"])
        assert "[1] Only passage" in prompt


# ---------------------------------------------------------------------------
# build_refusal_prompt
# ---------------------------------------------------------------------------

class TestBuildRefusalPrompt:
    """Tests for the build_refusal_prompt utility."""

    def test_includes_question(self):
        prompt = build_refusal_prompt("Why is the sky blue?")
        assert "Why is the sky blue?" in prompt


# ---------------------------------------------------------------------------
# answer_node — successful generation
# ---------------------------------------------------------------------------

class TestAnswerNodeSuccess:
    """Tests for answer_node with successful LLM generation."""

    def test_returns_answer(self):
        llm = MockLLM("Federalism is a system of government.")
        state = (
            StateBuilder()
            .with_question("What is federalism?")
            .with_search_results([_search_result("c-1", 0.9, "Federalism is...")])
            .build()
        )

        out = answer_node(state, llm)

        assert out["answer"] == "Federalism is a system of government."
        assert "error" not in out

    def test_confidence_from_scores(self):
        llm = MockLLM("Answer")
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([
                _search_result("c-1", 0.9),
                _search_result("c-2", 0.8),
            ])
            .build()
        )

        out = answer_node(state, llm)

        assert out["confidence"] == pytest.approx(0.85, abs=0.01)

    def test_prompt_contains_context_and_question(self):
        llm = MockLLM("Answer")
        state = (
            StateBuilder()
            .with_question("What is X?")
            .with_search_results([_search_result("c-1", 0.9, "X is a thing.")])
            .build()
        )

        answer_node(state, llm)

        assert "What is X?" in llm.last_prompt
        assert "X is a thing." in llm.last_prompt

    def test_multiple_contexts_in_prompt(self):
        llm = MockLLM("Answer")
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([
                _search_result("c-1", 0.9, "First passage"),
                _search_result("c-2", 0.8, "Second passage"),
            ])
            .build()
        )

        answer_node(state, llm)

        assert "First passage" in llm.last_prompt
        assert "Second passage" in llm.last_prompt

    def test_answer_is_stripped(self):
        llm = MockLLM("  padded answer  \n")
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([_search_result()])
            .build()
        )

        out = answer_node(state, llm)

        assert out["answer"] == "padded answer"

    def test_llm_called_once(self):
        llm = MockLLM("Answer")
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([_search_result()])
            .build()
        )

        answer_node(state, llm)

        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# answer_node — empty context / refusal
# ---------------------------------------------------------------------------

class TestAnswerNodeRefusal:
    """Tests for answer_node when there is no context."""

    def test_empty_results_triggers_refusal(self):
        llm = MockLLM("I'm sorry, I cannot answer that.")
        state = StateBuilder().with_question("Q").build()

        out = answer_node(state, llm)

        assert out["answer"] is None
        assert out["confidence"] == 0.0
        assert "refusal_reason" in out
        assert out["refusal_reason"] == "I'm sorry, I cannot answer that."

    def test_refusal_prompt_sent_to_llm(self):
        llm = MockLLM("Refusal text")
        state = StateBuilder().with_question("Unknown topic?").build()

        answer_node(state, llm)

        assert "Unknown topic?" in llm.last_prompt

    def test_refusal_on_missing_search_results_key(self):
        llm = MockLLM("Refusal")
        state: GraphState = {"question": "Q"}

        out = answer_node(state, llm)

        assert out["answer"] is None
        assert "refusal_reason" in out

    def test_refusal_llm_failure_has_default(self):
        llm = FailingLLM()
        state = StateBuilder().with_question("Q").build()

        out = answer_node(state, llm)

        assert out["answer"] is None
        assert "refusal_reason" in out
        assert "Insufficient context" in out["refusal_reason"]


# ---------------------------------------------------------------------------
# answer_node — error handling
# ---------------------------------------------------------------------------

class TestAnswerNodeErrors:
    """Tests for answer_node error handling."""

    def test_llm_failure_returns_error(self):
        llm = FailingLLM()
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([_search_result()])
            .build()
        )

        out = answer_node(state, llm)

        assert out["answer"] is None
        assert out["confidence"] == 0.0
        assert "error" in out
        assert "LLM generation failed" in out["error"]

    def test_empty_question(self):
        llm = MockLLM("Answer")
        state: GraphState = {"question": ""}

        out = answer_node(state, llm)

        assert out["answer"] is None
        assert "error" in out
        assert llm.call_count == 0

    def test_missing_question(self):
        llm = MockLLM("Answer")
        state: GraphState = {}  # type: ignore[typeddict-item]

        out = answer_node(state, llm)

        assert out["answer"] is None
        assert "error" in out

    def test_error_includes_exception_message(self):
        llm = FailingLLM()
        state = (
            StateBuilder()
            .with_question("Q")
            .with_search_results([_search_result()])
            .build()
        )

        out = answer_node(state, llm)

        assert "unavailable" in out["error"]


# ---------------------------------------------------------------------------
# BaseLLM protocol
# ---------------------------------------------------------------------------

class TestBaseLLMProtocol:
    """Tests that MockLLM satisfies the BaseLLM protocol."""

    def test_mock_llm_satisfies_protocol(self):
        assert isinstance(MockLLM(), BaseLLM)

    def test_failing_llm_satisfies_protocol(self):
        assert isinstance(FailingLLM(), BaseLLM)

    def test_random_object_does_not_satisfy(self):
        assert not isinstance("not an llm", BaseLLM)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify answer_node is importable from the nodes package."""

    def test_answer_node_exported(self):
        from src.graphs.nodes import answer_node as fn
        assert callable(fn)

    def test_prompts_exported(self):
        from src.graphs.prompts import RAG_PROMPT, REFUSAL_PROMPT
        assert RAG_PROMPT
        assert REFUSAL_PROMPT

    def test_builders_exported(self):
        from src.graphs.prompts import build_rag_prompt, build_refusal_prompt
        assert callable(build_rag_prompt)
        assert callable(build_refusal_prompt)
