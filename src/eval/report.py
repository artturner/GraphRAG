"""Evaluation report models.

This module defines the data structures for evaluation reports,
including per-question results and aggregate metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from typing import Any


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionResult:
    """Result of evaluating a single question.

    Attributes:
        question: The original question text.
        answer: The answer produced by the system (``None`` if refused).
        refusal_reason: Refusal reason if the system refused.
        groundedness: Groundedness score in [0, 1] (``None`` for refusals).
        relevance: Relevance score in [0, 1] (``None`` for refusals).
        refusal_correct: Whether the refusal decision was appropriate.
        expected_refusal: Whether a refusal was expected.
        latency_ms: Time to process the question in milliseconds.
        error: Error message if the question failed to execute.
    """

    question: str
    answer: str | None = None
    refusal_reason: str | None = None
    groundedness: float | None = None
    relevance: float | None = None
    refusal_correct: bool = False
    expected_refusal: bool = False
    latency_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate evaluation metrics across all questions.

    Attributes:
        total_questions: Total number of questions evaluated.
        answered_count: Number of questions that received an answer.
        refused_count: Number of questions the system refused.
        error_count: Number of questions that failed with errors.
        avg_groundedness: Mean groundedness score over answered questions.
        avg_relevance: Mean relevance score over answered questions.
        refusal_accuracy: Fraction of correct refusal decisions.
        avg_latency_ms: Mean latency across all questions.
    """

    total_questions: int = 0
    answered_count: int = 0
    refused_count: int = 0
    error_count: int = 0
    avg_groundedness: float = 0.0
    avg_relevance: float = 0.0
    refusal_accuracy: float = 0.0
    avg_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Full evaluation report.

    Attributes:
        suite_name: Name of the evaluation suite / dataset.
        created_at: Timestamp when the report was generated.
        metrics: Aggregate metrics computed from results.
        results: Per-question evaluation results.
    """

    suite_name: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    metrics: AggregateMetrics = field(default_factory=AggregateMetrics)
    results: list[QuestionResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Aggregate calculation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> None:
        """Recompute :attr:`metrics` from the current :attr:`results`."""
        total = len(self.results)
        if total == 0:
            self.metrics = AggregateMetrics()
            return

        answered = [r for r in self.results if r.answer is not None and r.error is None]
        refused = [r for r in self.results if r.refusal_reason is not None and r.error is None]
        errors = [r for r in self.results if r.error is not None]

        groundedness_scores = [
            r.groundedness for r in answered if r.groundedness is not None
        ]
        relevance_scores = [
            r.relevance for r in answered if r.relevance is not None
        ]
        refusal_decisions = [r for r in self.results if r.error is None]

        avg_g = (
            sum(groundedness_scores) / len(groundedness_scores)
            if groundedness_scores
            else 0.0
        )
        avg_r = (
            sum(relevance_scores) / len(relevance_scores)
            if relevance_scores
            else 0.0
        )
        refusal_acc = (
            sum(1 for r in refusal_decisions if r.refusal_correct)
            / len(refusal_decisions)
            if refusal_decisions
            else 0.0
        )
        avg_lat = (
            sum(r.latency_ms for r in self.results) / total
        )

        self.metrics = AggregateMetrics(
            total_questions=total,
            answered_count=len(answered),
            refused_count=len(refused),
            error_count=len(errors),
            avg_groundedness=round(avg_g, 4),
            avg_relevance=round(avg_r, 4),
            refusal_accuracy=round(refusal_acc, 4),
            avg_latency_ms=round(avg_lat, 2),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary."""
        return {
            "suite_name": self.suite_name,
            "created_at": self.created_at,
            "metrics": {
                "total_questions": self.metrics.total_questions,
                "answered_count": self.metrics.answered_count,
                "refused_count": self.metrics.refused_count,
                "error_count": self.metrics.error_count,
                "avg_groundedness": self.metrics.avg_groundedness,
                "avg_relevance": self.metrics.avg_relevance,
                "refusal_accuracy": self.metrics.refusal_accuracy,
                "avg_latency_ms": self.metrics.avg_latency_ms,
            },
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "refusal_reason": r.refusal_reason,
                    "groundedness": r.groundedness,
                    "relevance": r.relevance,
                    "refusal_correct": r.refusal_correct,
                    "expected_refusal": r.expected_refusal,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self) -> str:
        """Render the report as a self-contained HTML page."""
        m = self.metrics
        rows: list[str] = []
        for r in self.results:
            status = "error" if r.error else ("refused" if r.refusal_reason else "answered")
            g_str = f"{r.groundedness:.2f}" if r.groundedness is not None else "—"
            r_str = f"{r.relevance:.2f}" if r.relevance is not None else "—"
            ref_str = "Yes" if r.refusal_correct else "No"
            rows.append(
                f"<tr>"
                f"<td>{escape(r.question)}</td>"
                f"<td>{status}</td>"
                f"<td>{escape(r.answer or r.refusal_reason or r.error or '')}</td>"
                f"<td>{g_str}</td>"
                f"<td>{r_str}</td>"
                f"<td>{ref_str}</td>"
                f"<td>{r.latency_ms:.0f}</td>"
                f"</tr>"
            )

        table_body = "\n".join(rows)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Report — {escape(self.suite_name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  .metrics {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .metric {{ background: #f5f5f5; padding: 1rem 1.5rem; border-radius: 8px; }}
  .metric .value {{ font-size: 1.5rem; font-weight: bold; }}
  .metric .label {{ font-size: 0.85rem; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<p class="meta">Suite: <strong>{escape(self.suite_name)}</strong> &middot; {escape(self.created_at)}</p>

<div class="metrics">
  <div class="metric"><div class="value">{m.total_questions}</div><div class="label">Total Questions</div></div>
  <div class="metric"><div class="value">{m.avg_groundedness:.2f}</div><div class="label">Avg Groundedness</div></div>
  <div class="metric"><div class="value">{m.avg_relevance:.2f}</div><div class="label">Avg Relevance</div></div>
  <div class="metric"><div class="value">{m.refusal_accuracy:.0%}</div><div class="label">Refusal Accuracy</div></div>
  <div class="metric"><div class="value">{m.avg_latency_ms:.0f} ms</div><div class="label">Avg Latency</div></div>
</div>

<table>
<thead>
<tr><th>Question</th><th>Status</th><th>Answer / Reason</th><th>Ground.</th><th>Relev.</th><th>Ref. OK</th><th>Latency</th></tr>
</thead>
<tbody>
{table_body}
</tbody>
</table>
</body>
</html>"""
