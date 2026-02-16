"""Refusal correctness scorer for RAG evaluation.

This scorer checks whether the system's decision to answer or refuse
was appropriate given the evaluation question's expectations.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval.dataset import EvalQuestion
from src.types import QueryResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefusalResult:
    """Full output of the refusal scorer.

    Attributes:
        correct: Whether the refusal decision was appropriate.
        expected_refusal: Whether a refusal was expected.
        actual_refusal: Whether the system actually refused.
        reason: Human-readable explanation of the verdict.
    """

    correct: bool
    expected_refusal: bool
    actual_refusal: bool
    reason: str


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class RefusalScorer:
    """Score whether the system's refusal decision was correct.

    A refusal is detected when the :class:`~src.types.Answer` has a
    non-``None`` ``refusal_reason``.  The scorer compares this against
    the ``expected_refusal`` flag on the :class:`~src.eval.dataset.EvalQuestion`.

    There are four possible outcomes:

    * **True positive** – expected refusal, system refused → correct.
    * **True negative** – expected answer, system answered → correct.
    * **False positive** – expected answer, system refused → incorrect.
    * **False negative** – expected refusal, system answered → incorrect.

    Example:
        ```python
        scorer = RefusalScorer()
        is_correct = scorer.score(result, expected)
        ```
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: QueryResult, expected: EvalQuestion) -> bool:
        """Return whether the refusal decision was correct.

        Args:
            result: The query result produced by the RAG system.
            expected: The evaluation question with expected outcomes.

        Returns:
            ``True`` when the system's refusal/answer decision matches
            the expectation, ``False`` otherwise.
        """
        return self.score_detailed(result, expected).correct

    def score_detailed(
        self,
        result: QueryResult,
        expected: EvalQuestion,
    ) -> RefusalResult:
        """Return a detailed refusal-correctness result.

        Args:
            result: The query result produced by the RAG system.
            expected: The evaluation question with expected outcomes.

        Returns:
            A :class:`RefusalResult` with the verdict and explanation.
        """
        expected_refusal = expected.expected_refusal
        actual_refusal = result.answer.refusal_reason is not None

        if expected_refusal and actual_refusal:
            reason = "Correctly refused to answer."
        elif not expected_refusal and not actual_refusal:
            reason = "Correctly provided an answer."
        elif expected_refusal and not actual_refusal:
            reason = (
                "Should have refused but provided an answer."
            )
        else:
            reason = (
                "Should have answered but refused: "
                f"{result.answer.refusal_reason}"
            )

        return RefusalResult(
            correct=(expected_refusal == actual_refusal),
            expected_refusal=expected_refusal,
            actual_refusal=actual_refusal,
            reason=reason,
        )
