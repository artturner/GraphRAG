"""Tests for the refusal correctness scorer."""

from __future__ import annotations

import pytest

from src.eval.dataset import EvalQuestion
from src.eval.scorers import RefusalResult, RefusalScorer
from src.types import Answer, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _answered_result(question: str = "What is X?", text: str = "X is Y.") -> QueryResult:
    """Create a QueryResult where the system provided an answer."""
    return QueryResult(
        question=question,
        answer=Answer(text=text, confidence=0.9),
        mode="vector",
        latency_ms=100.0,
    )


def _refused_result(
    question: str = "What is X?",
    refusal_reason: str = "Not in corpus",
) -> QueryResult:
    """Create a QueryResult where the system refused to answer."""
    return QueryResult(
        question=question,
        answer=Answer(
            text="",
            confidence=0.0,
            refusal_reason=refusal_reason,
        ),
        mode="vector",
        latency_ms=50.0,
    )


def _expected_answer(question: str = "What is X?") -> EvalQuestion:
    """Create an EvalQuestion that expects an answer."""
    return EvalQuestion(question=question)


def _expected_refusal(
    question: str = "What is the capital of Mars?",
    reason: str = "not in corpus",
) -> EvalQuestion:
    """Create an EvalQuestion that expects a refusal."""
    return EvalQuestion(
        question=question,
        expected_refusal=True,
        refusal_reason=reason,
    )


# ---------------------------------------------------------------------------
# score() — correct refusal (true positive)
# ---------------------------------------------------------------------------


class TestCorrectRefusal:
    """Tests where a refusal was expected and the system refused."""

    def test_basic_correct_refusal(self):
        scorer = RefusalScorer()
        result = _refused_result()
        expected = _expected_refusal()
        assert scorer.score(result, expected) is True

    def test_correct_refusal_various_reasons(self):
        scorer = RefusalScorer()
        for reason in ("Not in corpus", "Insufficient context", "Out of scope"):
            result = _refused_result(refusal_reason=reason)
            expected = _expected_refusal()
            assert scorer.score(result, expected) is True

    def test_correct_refusal_detailed(self):
        scorer = RefusalScorer()
        result = _refused_result()
        expected = _expected_refusal()
        detail = scorer.score_detailed(result, expected)
        assert detail.correct is True
        assert detail.expected_refusal is True
        assert detail.actual_refusal is True
        assert "Correctly refused" in detail.reason


# ---------------------------------------------------------------------------
# score() — correct answer (true negative)
# ---------------------------------------------------------------------------


class TestCorrectAnswer:
    """Tests where an answer was expected and the system answered."""

    def test_basic_correct_answer(self):
        scorer = RefusalScorer()
        result = _answered_result()
        expected = _expected_answer()
        assert scorer.score(result, expected) is True

    def test_correct_answer_with_citations(self):
        scorer = RefusalScorer()
        result = _answered_result(text="Federalism is a system of government.")
        expected = EvalQuestion(
            question="What is federalism?",
            expected_answer_contains=["system"],
            expected_citations_min=1,
        )
        assert scorer.score(result, expected) is True

    def test_correct_answer_detailed(self):
        scorer = RefusalScorer()
        result = _answered_result()
        expected = _expected_answer()
        detail = scorer.score_detailed(result, expected)
        assert detail.correct is True
        assert detail.expected_refusal is False
        assert detail.actual_refusal is False
        assert "Correctly provided" in detail.reason


# ---------------------------------------------------------------------------
# score() — incorrect refusal (false positive: should have answered)
# ---------------------------------------------------------------------------


class TestIncorrectRefusal:
    """Tests where the system refused but should have answered."""

    def test_refused_when_answer_expected(self):
        scorer = RefusalScorer()
        result = _refused_result(refusal_reason="No relevant documents found")
        expected = _expected_answer()
        assert scorer.score(result, expected) is False

    def test_incorrect_refusal_detailed(self):
        scorer = RefusalScorer()
        result = _refused_result(refusal_reason="Insufficient context")
        expected = _expected_answer()
        detail = scorer.score_detailed(result, expected)
        assert detail.correct is False
        assert detail.expected_refusal is False
        assert detail.actual_refusal is True
        assert "Should have answered but refused" in detail.reason
        assert "Insufficient context" in detail.reason

    def test_incorrect_refusal_includes_system_reason(self):
        scorer = RefusalScorer()
        reason = "Query out of scope"
        result = _refused_result(refusal_reason=reason)
        expected = _expected_answer()
        detail = scorer.score_detailed(result, expected)
        assert reason in detail.reason


# ---------------------------------------------------------------------------
# score() — incorrect answer (false negative: should have refused)
# ---------------------------------------------------------------------------


class TestIncorrectAnswer:
    """Tests where the system answered but should have refused."""

    def test_answered_when_refusal_expected(self):
        scorer = RefusalScorer()
        result = _answered_result(text="Mars capital is Olympus Mons.")
        expected = _expected_refusal()
        assert scorer.score(result, expected) is False

    def test_incorrect_answer_detailed(self):
        scorer = RefusalScorer()
        result = _answered_result(text="Some hallucinated answer.")
        expected = _expected_refusal()
        detail = scorer.score_detailed(result, expected)
        assert detail.correct is False
        assert detail.expected_refusal is True
        assert detail.actual_refusal is False
        assert "Should have refused" in detail.reason

    def test_incorrect_answer_with_high_confidence(self):
        scorer = RefusalScorer()
        result = QueryResult(
            question="What is the capital of Mars?",
            answer=Answer(text="Olympus Mons", confidence=0.99),
            mode="hybrid",
            latency_ms=200.0,
        )
        expected = _expected_refusal()
        assert scorer.score(result, expected) is False


# ---------------------------------------------------------------------------
# score_detailed() — result structure
# ---------------------------------------------------------------------------


class TestScoreDetailed:
    """Tests for the detailed scoring result."""

    def test_returns_refusal_result(self):
        scorer = RefusalScorer()
        result = _answered_result()
        expected = _expected_answer()
        detail = scorer.score_detailed(result, expected)
        assert isinstance(detail, RefusalResult)

    def test_score_matches_score_detailed(self):
        scorer = RefusalScorer()
        result = _refused_result()
        expected = _expected_refusal()
        assert scorer.score(result, expected) == scorer.score_detailed(
            result, expected
        ).correct

    def test_all_four_outcomes(self):
        """Verify each of the four TP/TN/FP/FN outcomes."""
        scorer = RefusalScorer()

        # True positive
        tp = scorer.score_detailed(_refused_result(), _expected_refusal())
        assert tp.correct is True and tp.expected_refusal and tp.actual_refusal

        # True negative
        tn = scorer.score_detailed(_answered_result(), _expected_answer())
        assert tn.correct is True and not tn.expected_refusal and not tn.actual_refusal

        # False positive (refused but shouldn't have)
        fp = scorer.score_detailed(_refused_result(), _expected_answer())
        assert fp.correct is False and not fp.expected_refusal and fp.actual_refusal

        # False negative (answered but shouldn't have)
        fn = scorer.score_detailed(_answered_result(), _expected_refusal())
        assert fn.correct is False and fn.expected_refusal and not fn.actual_refusal


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests."""

    def test_empty_refusal_reason_is_not_refusal(self):
        """refusal_reason=None means answered, not refused."""
        scorer = RefusalScorer()
        result = QueryResult(
            question="Q?",
            answer=Answer(text="Answer.", confidence=0.5, refusal_reason=None),
            mode="vector",
            latency_ms=10.0,
        )
        expected = _expected_answer()
        assert scorer.score(result, expected) is True

    def test_empty_string_answer_with_refusal_reason(self):
        """An empty text with refusal_reason set counts as refusal."""
        scorer = RefusalScorer()
        result = QueryResult(
            question="Q?",
            answer=Answer(text="", confidence=0.0, refusal_reason="Cannot answer"),
            mode="vector",
            latency_ms=10.0,
        )
        expected = _expected_refusal(question="Q?")
        assert scorer.score(result, expected) is True

    def test_different_retrieval_modes(self):
        """Scorer works regardless of retrieval mode."""
        scorer = RefusalScorer()
        for mode in ("vector", "graph", "hybrid"):
            result = QueryResult(
                question="Q?",
                answer=Answer(text="A.", confidence=0.8),
                mode=mode,
                latency_ms=50.0,
            )
            expected = _expected_answer(question="Q?")
            assert scorer.score(result, expected) is True


# ---------------------------------------------------------------------------
# Data class immutability
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for result dataclass properties."""

    def test_refusal_result_frozen(self):
        r = RefusalResult(
            correct=True,
            expected_refusal=True,
            actual_refusal=True,
            reason="test",
        )
        with pytest.raises(AttributeError):
            r.correct = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify scorer is importable from the scorers package."""

    def test_scorer_exported(self):
        from src.eval.scorers import RefusalScorer as cls
        assert cls is not None

    def test_result_exported(self):
        from src.eval.scorers import RefusalResult
        assert RefusalResult is not None
