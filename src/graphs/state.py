"""Graph state definition for the LangGraph Q&A workflow.

This module defines the TypedDict that flows through every node in
the LangGraph workflow, along with helpers for constructing and
validating state objects.
"""

from typing import TypedDict

from src.store.base import SearchResult
from src.types import Chunk, Citation


class GraphState(TypedDict, total=False):
    """State that flows through the LangGraph Q&A workflow.

    All fields except ``question`` are optional so that nodes can
    progressively populate the state as the workflow advances.

    Attributes:
        question: The user's original question (required to start).
        query_type: Classification of the query (factual, procedural, unsupported).
        chunks: Raw document chunks retrieved from the store.
        search_results: Scored search results from the retrieval step.
        answer: Generated answer text, or ``None`` if not yet produced.
        citations: Citations supporting the answer.
        confidence: Confidence / groundedness score in [0, 1].
        retry_count: Number of answer-generation retries so far.
        refusal_reason: Human-readable reason if the system refused.
        error: Error message if a node failed.

    Example:
        ```python
        state: GraphState = {
            "question": "What is federalism?",
            "query_type": "factual",
            "chunks": [],
            "search_results": [],
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "retry_count": 0,
            "refusal_reason": None,
            "error": None,
        }
        ```
    """

    question: str
    query_type: str
    chunks: list[Chunk]
    search_results: list[SearchResult]
    answer: str | None
    citations: list[Citation]
    confidence: float
    retry_count: int
    refusal_reason: str | None
    error: str | None


# -- helpers ---------------------------------------------------------------


class StateBuilder:
    """Fluent builder for constructing an initial ``GraphState``.

    Every field has a sensible default so callers only need to supply
    the question (at minimum).

    Example:
        ```python
        state = (
            StateBuilder()
            .with_question("What is federalism?")
            .build()
        )
        ```
    """

    def __init__(self) -> None:
        self._question: str | None = None
        self._query_type: str | None = None
        self._chunks: list[Chunk] = []
        self._search_results: list[SearchResult] = []
        self._answer: str | None = None
        self._citations: list[Citation] = []
        self._confidence: float = 0.0
        self._retry_count: int = 0
        self._refusal_reason: str | None = None
        self._error: str | None = None

    def with_question(self, question: str) -> "StateBuilder":
        """Set the user question."""
        self._question = question
        return self

    def with_query_type(self, query_type: str) -> "StateBuilder":
        """Set the query type classification."""
        self._query_type = query_type
        return self

    def with_chunks(self, chunks: list[Chunk]) -> "StateBuilder":
        """Set the retrieved chunks."""
        self._chunks = chunks
        return self

    def with_search_results(self, results: list[SearchResult]) -> "StateBuilder":
        """Set the search results."""
        self._search_results = results
        return self

    def with_answer(self, answer: str) -> "StateBuilder":
        """Set the generated answer."""
        self._answer = answer
        return self

    def with_citations(self, citations: list[Citation]) -> "StateBuilder":
        """Set the citations."""
        self._citations = citations
        return self

    def with_confidence(self, confidence: float) -> "StateBuilder":
        """Set the confidence score."""
        self._confidence = confidence
        return self

    def with_retry_count(self, count: int) -> "StateBuilder":
        """Set the retry count."""
        self._retry_count = count
        return self

    def with_refusal_reason(self, reason: str) -> "StateBuilder":
        """Set a refusal reason."""
        self._refusal_reason = reason
        return self

    def with_error(self, error: str) -> "StateBuilder":
        """Set an error message."""
        self._error = error
        return self

    def build(self) -> GraphState:
        """Build and return the ``GraphState``.

        Raises:
            ValueError: If ``question`` has not been set.
        """
        if self._question is None:
            raise ValueError("question is required to build a GraphState")

        state = GraphState(
            question=self._question,
            chunks=self._chunks,
            search_results=self._search_results,
            answer=self._answer,
            citations=self._citations,
            confidence=self._confidence,
            retry_count=self._retry_count,
            refusal_reason=self._refusal_reason,
            error=self._error,
        )
        if self._query_type is not None:
            state["query_type"] = self._query_type
        return state


class StateValidator:
    """Validates ``GraphState`` dicts at workflow-node boundaries.

    Each ``validate_*`` method checks the preconditions that must hold
    before the corresponding graph node executes.

    Example:
        ```python
        validator = StateValidator()
        errors = validator.validate_for_retrieval(state)
        if errors:
            raise ValueError(errors)
        ```
    """

    def validate_initial(self, state: GraphState) -> list[str]:
        """Validate that a state is suitable to start the workflow.

        Requirements:
        - ``question`` must be a non-empty string.
        """
        errors: list[str] = []
        question = state.get("question")
        if not question or not isinstance(question, str) or not question.strip():
            errors.append("question must be a non-empty string")
        return errors

    def validate_for_retrieval(self, state: GraphState) -> list[str]:
        """Validate state before the retrieval node.

        Requirements:
        - passes ``validate_initial``.
        """
        return self.validate_initial(state)

    def validate_for_answer(self, state: GraphState) -> list[str]:
        """Validate state before the answer-generation node.

        Requirements:
        - passes ``validate_initial``.
        - ``search_results`` must be present (may be empty).
        """
        errors = self.validate_initial(state)
        if "search_results" not in state:
            errors.append("search_results must be present before answer generation")
        return errors

    def validate_for_grounding(self, state: GraphState) -> list[str]:
        """Validate state before the grounding-verification node.

        Requirements:
        - passes ``validate_for_answer``.
        - ``answer`` must be a non-empty string.
        """
        errors = self.validate_for_answer(state)
        answer = state.get("answer")
        if not answer or not isinstance(answer, str) or not answer.strip():
            errors.append("answer must be a non-empty string for grounding check")
        return errors

    def validate_complete(self, state: GraphState) -> list[str]:
        """Validate that a state represents a completed workflow run.

        A completed state must have either:
        - a non-empty ``answer`` with ``confidence`` > 0, or
        - a non-empty ``refusal_reason``, or
        - a non-empty ``error``.
        """
        errors: list[str] = []
        answer = state.get("answer")
        refusal = state.get("refusal_reason")
        error = state.get("error")

        has_answer = answer and isinstance(answer, str) and answer.strip()
        has_refusal = refusal and isinstance(refusal, str) and refusal.strip()
        has_error = error and isinstance(error, str) and error.strip()

        if not (has_answer or has_refusal or has_error):
            errors.append(
                "completed state must have an answer, refusal_reason, or error"
            )

        if has_answer:
            confidence = state.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                errors.append("confidence must be a float between 0.0 and 1.0")

        return errors
