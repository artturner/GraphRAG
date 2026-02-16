"""Evaluation runner for the GraphRAG system.

This module runs an evaluation dataset through a compiled LangGraph
Q&A workflow, collects per-question scores using the evaluation
scorers, and produces an :class:`~src.eval.report.EvalReport`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, Sequence, runtime_checkable

from src.eval.dataset import EvalDataset, EvalQuestion
from src.eval.report import AggregateMetrics, EvalReport, QuestionResult
from src.eval.scorers.groundedness import GroundednessScorer
from src.eval.scorers.refusal import RefusalScorer
from src.eval.scorers.relevance import RelevanceScorer
from src.types import Answer, Chunk, Citation, QueryResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph protocol — any object with an ``invoke`` method
# ---------------------------------------------------------------------------


@runtime_checkable
class Invocable(Protocol):
    """Minimal protocol for a compiled LangGraph graph."""

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Run an evaluation dataset through a graph and score results.

    The runner iterates over every question in the dataset, invokes the
    graph, converts the raw state dict to a :class:`~src.types.QueryResult`,
    scores each result with the groundedness, relevance, and refusal
    scorers, and finally compiles an :class:`~src.eval.report.EvalReport`.

    Args:
        groundedness_scorer: Scorer for answer groundedness.
        relevance_scorer: Scorer for question–answer relevance.
        refusal_scorer: Scorer for refusal correctness.

    Example:
        ```python
        runner = EvalRunner()
        report = runner.run(dataset, graph)
        print(report.metrics.avg_groundedness)
        ```
    """

    def __init__(
        self,
        *,
        groundedness_scorer: GroundednessScorer | None = None,
        relevance_scorer: RelevanceScorer | None = None,
        refusal_scorer: RefusalScorer | None = None,
    ) -> None:
        self._groundedness = groundedness_scorer or GroundednessScorer()
        self._relevance = relevance_scorer or RelevanceScorer()
        self._refusal = refusal_scorer or RefusalScorer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: EvalDataset,
        graph: Invocable,
        *,
        mode: str = "vector",
    ) -> EvalReport:
        """Evaluate every question in *dataset* using *graph*.

        Args:
            dataset: The evaluation dataset.
            graph: A compiled LangGraph (or any object with an
                ``invoke(dict) -> dict`` method).
            mode: Retrieval mode label written into QueryResult
                (``"vector"``, ``"graph"``, or ``"hybrid"``).

        Returns:
            An :class:`EvalReport` with per-question results and
            aggregate metrics.
        """
        report = EvalReport(suite_name=dataset.name)

        for eq in dataset.questions:
            qr = self._evaluate_question(eq, graph, mode=mode)
            report.results.append(qr)

        report.compute_metrics()
        logger.info(
            "Evaluation complete: %d questions, avg_groundedness=%.2f, "
            "avg_relevance=%.2f, refusal_accuracy=%.2f",
            report.metrics.total_questions,
            report.metrics.avg_groundedness,
            report.metrics.avg_relevance,
            report.metrics.refusal_accuracy,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_question(
        self,
        eq: EvalQuestion,
        graph: Invocable,
        *,
        mode: str,
    ) -> QuestionResult:
        """Run a single question through the graph and score it."""
        start = time.perf_counter()

        try:
            state = graph.invoke({"question": eq.question})
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Question failed: %s — %s", eq.question, exc)
            return QuestionResult(
                question=eq.question,
                expected_refusal=eq.expected_refusal,
                latency_ms=round(elapsed, 2),
                error=str(exc),
            )

        elapsed = (time.perf_counter() - start) * 1000

        # -- Convert state dict → QueryResult --------------------------
        query_result = self._state_to_query_result(state, mode=mode)

        # -- Score -----------------------------------------------------
        is_refusal = query_result.answer.refusal_reason is not None
        refusal_correct = self._refusal.score(query_result, eq)

        groundedness: float | None = None
        relevance: float | None = None

        if not is_refusal:
            chunks = self._extract_chunks(state)
            groundedness = self._groundedness.score(
                query_result.answer.text, chunks,
            )
            relevance = self._relevance.score(
                query_result.answer.text, eq.question,
            )

        return QuestionResult(
            question=eq.question,
            answer=query_result.answer.text if not is_refusal else None,
            refusal_reason=query_result.answer.refusal_reason,
            groundedness=groundedness,
            relevance=relevance,
            refusal_correct=refusal_correct,
            expected_refusal=eq.expected_refusal,
            latency_ms=round(elapsed, 2),
        )

    @staticmethod
    def _state_to_query_result(
        state: dict[str, Any],
        *,
        mode: str,
    ) -> QueryResult:
        """Convert a raw graph state dict into a QueryResult."""
        answer_text = state.get("answer") or ""
        citations = state.get("citations") or []
        confidence = state.get("confidence", 0.0)
        refusal_reason = state.get("refusal_reason")

        answer = Answer(
            text=answer_text,
            citations=citations,
            confidence=max(0.0, min(1.0, confidence)),
            refusal_reason=refusal_reason,
        )

        return QueryResult(
            question=state.get("question", ""),
            answer=answer,
            mode=mode,
            latency_ms=0.0,  # real latency tracked separately
        )

    @staticmethod
    def _extract_chunks(state: dict[str, Any]) -> list[Chunk]:
        """Pull Chunk objects from the graph state."""
        chunks = state.get("chunks")
        if chunks and isinstance(chunks, list):
            return [c for c in chunks if isinstance(c, Chunk)]
        return []
