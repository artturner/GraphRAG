"""Tests for the route node query classifier."""

import pytest

from src.graphs.nodes import route_node
from src.graphs.state import GraphState, StateBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route(question: str) -> str:
    """Shortcut: run route_node and return the query_type."""
    state: GraphState = {"question": question}
    return route_node(state)["query_type"]


# ---------------------------------------------------------------------------
# Factual queries
# ---------------------------------------------------------------------------

class TestFactualRouting:
    """Queries that should be classified as 'factual'."""

    @pytest.mark.parametrize(
        "question",
        [
            "What is federalism?",
            "Who wrote the Constitution?",
            "When was the Declaration of Independence signed?",
            "Where is the Supreme Court located?",
            "Which amendment guarantees free speech?",
            "Why did the Civil War start?",
            "Is the Senate part of Congress?",
            "Are there term limits for senators?",
            "Was George Washington the first president?",
            "Does the president appoint judges?",
            "Do states have their own constitutions?",
            "Define separation of powers",
            "Describe the electoral college",
            "What does bicameral mean?",
            "Tell me about the Bill of Rights",
            "Explain the concept of federalism",
            "What is the meaning of due process?",
        ],
    )
    def test_factual_queries(self, question: str):
        assert _route(question) == "factual"

    def test_question_mark_fallback(self):
        """A question ending with '?' that doesn't match specific patterns."""
        assert _route("Federalism?") == "factual"


# ---------------------------------------------------------------------------
# Procedural queries
# ---------------------------------------------------------------------------

class TestProceduralRouting:
    """Queries that should be classified as 'procedural'."""

    @pytest.mark.parametrize(
        "question",
        [
            "How do I amend the Constitution?",
            "How can a bill become a law?",
            "How to register to vote?",
            "How should I prepare for the citizenship test?",
            "How would you impeach a president?",
            "How could I challenge a law?",
            "Steps to ratify an amendment",
            "What is the process of electing a president?",
            "Explain how a bill becomes a law",
            "Guide to understanding the judicial branch",
            "Instructions for filing a petition",
            "Procedure for appointing a Supreme Court justice",
        ],
    )
    def test_procedural_queries(self, question: str):
        assert _route(question) == "procedural"


# ---------------------------------------------------------------------------
# Unsupported queries
# ---------------------------------------------------------------------------

class TestUnsupportedRouting:
    """Queries that should be classified as 'unsupported'."""

    @pytest.mark.parametrize(
        "question",
        [
            "Hello",
            "Hi there",
            "Hey",
            "Good morning",
            "Good afternoon",
            "Thanks",
            "Thank you",
            "Bye",
            "Goodbye",
            "Ok",
            "Yes",
            "No",
            "Lol",
            "Haha",
            "Hmm",
            "Help",
            "Stop",
            "Quit",
        ],
    )
    def test_unsupported_queries(self, question: str):
        assert _route(question) == "unsupported"

    def test_plain_statement_without_question_mark(self):
        """A bare statement that doesn't match any pattern."""
        assert _route("pineapple pizza") == "unsupported"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_question(self):
        assert _route("") == "unsupported"

    def test_whitespace_only(self):
        assert _route("   ") == "unsupported"

    def test_missing_question_key(self):
        result = route_node({})  # type: ignore[typeddict-item]
        assert result["query_type"] == "unsupported"

    def test_case_insensitive(self):
        assert _route("WHAT IS FEDERALISM?") == "factual"
        assert _route("HOW DO I VOTE?") == "procedural"
        assert _route("HELLO") == "unsupported"

    def test_leading_whitespace_stripped(self):
        assert _route("  What is democracy?") == "factual"

    def test_trailing_whitespace_stripped(self):
        assert _route("What is democracy?   ") == "factual"

    def test_returns_dict_with_query_type_key(self):
        state: GraphState = {"question": "What is X?"}
        result = route_node(state)
        assert isinstance(result, dict)
        assert "query_type" in result

    def test_procedural_takes_precedence_over_factual(self):
        """'How do ...' should be procedural even though 'do' is factual."""
        assert _route("How do I file taxes?") == "procedural"

    def test_unsupported_takes_precedence_over_factual(self):
        """Greetings should remain unsupported regardless of question words."""
        assert _route("Hello") == "unsupported"

    def test_works_with_state_builder(self):
        state = StateBuilder().with_question("What is X?").build()
        result = route_node(state)
        assert result["query_type"] == "factual"

    def test_explain_how_is_procedural(self):
        assert _route("Explain how photosynthesis works") == "procedural"

    def test_explain_what_is_factual(self):
        assert _route("Explain what photosynthesis is") == "factual"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify route_node is importable from the nodes package."""

    def test_route_node_exported(self):
        from src.graphs.nodes import route_node as fn
        assert callable(fn)
