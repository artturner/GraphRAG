"""Tests for the evaluation runner and report generation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.eval import (
    AggregateMetrics,
    EvalDataset,
    EvalQuestion,
    EvalReport,
    EvalRunner,
    QuestionResult,
)
from src.types import Chunk, Citation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(content: str, chunk_id: str = "c-1") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={},
    )


class FakeGraph:
    """A fake compiled graph that returns scripted states."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        """Map question text → state dict to return."""
        self._responses = responses

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        question = input["question"]
        if question in self._responses:
            state = {"question": question}
            state.update(self._responses[question])
            return state
        raise ValueError(f"Unexpected question: {question}")


class ErrorGraph:
    """A graph that raises on every invocation."""

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Graph exploded")


def _make_dataset(*questions: EvalQuestion) -> EvalDataset:
    return EvalDataset(name="test_suite", questions=list(questions))


# -- Common state dicts ----------------------------------------------------

def _answered_state(
    answer: str = "Federalism is a system of government.",
    confidence: float = 0.95,
    content: str = "Federalism is a system of government.",
) -> dict[str, Any]:
    return {
        "answer": answer,
        "confidence": confidence,
        "citations": [
            Citation(source="doc.txt", chunk_id="c-1", text="cited", score=0.9),
        ],
        "chunks": [_chunk(content)],
        "refusal_reason": None,
    }


def _refused_state(reason: str = "Not in corpus") -> dict[str, Any]:
    return {
        "answer": None,
        "confidence": 0.0,
        "citations": [],
        "chunks": [],
        "refusal_reason": reason,
    }


# ---------------------------------------------------------------------------
# EvalRunner.run — happy path
# ---------------------------------------------------------------------------


class TestRunnerHappyPath:
    """Tests where all questions execute without errors."""

    def test_single_answerable_question(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
        })
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        assert report.suite_name == "test_suite"
        assert len(report.results) == 1
        assert report.results[0].answer is not None
        assert report.results[0].error is None
        assert report.metrics.total_questions == 1
        assert report.metrics.answered_count == 1
        assert report.metrics.refused_count == 0

    def test_single_refusal_question(self):
        dataset = _make_dataset(
            EvalQuestion(
                question="What is the capital of France?",
                expected_refusal=True,
            ),
        )
        graph = FakeGraph({
            "What is the capital of France?": _refused_state(),
        })
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        assert len(report.results) == 1
        r = report.results[0]
        assert r.answer is None
        assert r.refusal_reason == "Not in corpus"
        assert r.refusal_correct is True
        assert r.groundedness is None
        assert r.relevance is None
        assert report.metrics.refused_count == 1

    def test_mixed_questions(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
            EvalQuestion(
                question="What is the capital of France?",
                expected_refusal=True,
            ),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
            "What is the capital of France?": _refused_state(),
        })
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        assert report.metrics.total_questions == 2
        assert report.metrics.answered_count == 1
        assert report.metrics.refused_count == 1
        assert report.metrics.refusal_accuracy == 1.0

    def test_groundedness_scored_for_answers(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
        })
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        r = report.results[0]
        assert r.groundedness is not None
        assert 0.0 <= r.groundedness <= 1.0

    def test_relevance_scored_for_answers(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
        })
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        r = report.results[0]
        assert r.relevance is not None
        assert 0.0 <= r.relevance <= 1.0

    def test_latency_recorded(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
        })
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        assert report.results[0].latency_ms >= 0.0
        assert report.metrics.avg_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# EvalRunner.run — error handling
# ---------------------------------------------------------------------------


class TestRunnerErrors:
    """Tests where graph invocation fails."""

    def test_graph_error_captured(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = ErrorGraph()
        runner = EvalRunner()
        report = runner.run(dataset, graph)

        assert len(report.results) == 1
        r = report.results[0]
        assert r.error is not None
        assert "exploded" in r.error
        assert report.metrics.error_count == 1

    def test_error_does_not_stop_remaining(self):
        """A failing question should not prevent the next question."""

        class PartialGraph:
            def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
                q = input["question"]
                if q == "fail":
                    raise RuntimeError("boom")
                return {"question": q, **_answered_state()}

        dataset = _make_dataset(
            EvalQuestion(question="fail"),
            EvalQuestion(question="What is federalism?"),
        )
        runner = EvalRunner()
        report = runner.run(dataset, PartialGraph())

        assert len(report.results) == 2
        assert report.results[0].error is not None
        assert report.results[1].error is None
        assert report.metrics.error_count == 1
        assert report.metrics.answered_count == 1


# ---------------------------------------------------------------------------
# EvalRunner — refusal accuracy
# ---------------------------------------------------------------------------


class TestRefusalAccuracy:
    """Tests focusing on the refusal correctness metric."""

    def test_all_correct(self):
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
            EvalQuestion(question="Capital of France?", expected_refusal=True),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
            "Capital of France?": _refused_state(),
        })
        report = EvalRunner().run(dataset, graph)
        assert report.metrics.refusal_accuracy == 1.0

    def test_false_negative_refusal(self):
        """System answered but should have refused."""
        dataset = _make_dataset(
            EvalQuestion(question="Capital of France?", expected_refusal=True),
        )
        graph = FakeGraph({
            "Capital of France?": _answered_state(answer="Paris."),
        })
        report = EvalRunner().run(dataset, graph)
        assert report.results[0].refusal_correct is False
        assert report.metrics.refusal_accuracy == 0.0

    def test_false_positive_refusal(self):
        """System refused but should have answered."""
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = FakeGraph({
            "What is federalism?": _refused_state("Not in corpus"),
        })
        report = EvalRunner().run(dataset, graph)
        assert report.results[0].refusal_correct is False
        assert report.metrics.refusal_accuracy == 0.0


# ---------------------------------------------------------------------------
# EvalRunner — custom scorers
# ---------------------------------------------------------------------------


class TestCustomScorers:
    """Tests for injecting custom scorer instances."""

    def test_custom_groundedness_threshold(self):
        from src.eval.scorers import GroundednessScorer

        runner = EvalRunner(groundedness_scorer=GroundednessScorer(threshold=0.01))
        dataset = _make_dataset(
            EvalQuestion(question="What is federalism?"),
        )
        graph = FakeGraph({
            "What is federalism?": _answered_state(),
        })
        report = runner.run(dataset, graph)
        assert report.results[0].groundedness is not None


# ---------------------------------------------------------------------------
# EvalReport — compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for the report metrics computation."""

    def test_empty_results(self):
        report = EvalReport(suite_name="empty")
        report.compute_metrics()
        assert report.metrics.total_questions == 0
        assert report.metrics.avg_groundedness == 0.0

    def test_recompute_updates_metrics(self):
        report = EvalReport(suite_name="test")
        report.results.append(
            QuestionResult(
                question="Q?",
                answer="A.",
                groundedness=0.8,
                relevance=0.9,
                refusal_correct=True,
                latency_ms=100.0,
            ),
        )
        report.compute_metrics()
        assert report.metrics.total_questions == 1
        assert report.metrics.avg_groundedness == 0.8
        assert report.metrics.avg_relevance == 0.9
        assert report.metrics.refusal_accuracy == 1.0
        assert report.metrics.avg_latency_ms == 100.0

    def test_mixed_results_metrics(self):
        report = EvalReport(suite_name="mixed")
        report.results = [
            QuestionResult(
                question="Q1?",
                answer="A1.",
                groundedness=1.0,
                relevance=0.8,
                refusal_correct=True,
                latency_ms=100.0,
            ),
            QuestionResult(
                question="Q2?",
                refusal_reason="Not in corpus",
                refusal_correct=True,
                expected_refusal=True,
                latency_ms=50.0,
            ),
            QuestionResult(
                question="Q3?",
                error="boom",
                latency_ms=10.0,
            ),
        ]
        report.compute_metrics()
        assert report.metrics.total_questions == 3
        assert report.metrics.answered_count == 1
        assert report.metrics.refused_count == 1
        assert report.metrics.error_count == 1
        assert report.metrics.avg_groundedness == 1.0
        assert report.metrics.avg_relevance == 0.8
        # refusal_accuracy: 2 non-error, both correct → 1.0
        assert report.metrics.refusal_accuracy == 1.0
        assert report.metrics.avg_latency_ms == pytest.approx(
            (100 + 50 + 10) / 3, abs=0.1,
        )


# ---------------------------------------------------------------------------
# EvalReport — to_json
# ---------------------------------------------------------------------------


class TestReportJSON:
    """Tests for JSON serialisation."""

    def test_to_json_valid(self):
        report = EvalReport(suite_name="json_test")
        report.results.append(
            QuestionResult(
                question="Q?",
                answer="A.",
                groundedness=0.9,
                relevance=0.8,
                refusal_correct=True,
                latency_ms=42.0,
            ),
        )
        report.compute_metrics()

        raw = report.to_json()
        data = json.loads(raw)

        assert data["suite_name"] == "json_test"
        assert data["metrics"]["total_questions"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["question"] == "Q?"

    def test_to_json_round_trip(self):
        report = EvalReport(suite_name="rt")
        report.results.append(
            QuestionResult(question="Q?", answer="A.", refusal_correct=True),
        )
        report.compute_metrics()

        data = json.loads(report.to_json())
        assert data == report.to_dict()

    def test_to_json_empty_report(self):
        report = EvalReport(suite_name="empty")
        report.compute_metrics()
        data = json.loads(report.to_json())
        assert data["metrics"]["total_questions"] == 0
        assert data["results"] == []

    def test_refusal_in_json(self):
        report = EvalReport(suite_name="ref")
        report.results.append(
            QuestionResult(
                question="Q?",
                refusal_reason="Out of scope",
                refusal_correct=True,
                expected_refusal=True,
            ),
        )
        report.compute_metrics()
        data = json.loads(report.to_json())
        assert data["results"][0]["refusal_reason"] == "Out of scope"
        assert data["results"][0]["answer"] is None


# ---------------------------------------------------------------------------
# EvalReport — to_html
# ---------------------------------------------------------------------------


class TestReportHTML:
    """Tests for HTML report generation."""

    def test_to_html_contains_suite_name(self):
        report = EvalReport(suite_name="html_test")
        report.compute_metrics()
        html = report.to_html()
        assert "html_test" in html

    def test_to_html_contains_question(self):
        report = EvalReport(suite_name="html")
        report.results.append(
            QuestionResult(question="What is X?", answer="X is Y."),
        )
        report.compute_metrics()
        html = report.to_html()
        assert "What is X?" in html

    def test_to_html_valid_structure(self):
        report = EvalReport(suite_name="struct")
        report.results.append(
            QuestionResult(
                question="Q?",
                answer="A.",
                groundedness=0.9,
                relevance=0.8,
                refusal_correct=True,
                latency_ms=100.0,
            ),
        )
        report.compute_metrics()
        html = report.to_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "<table>" in html
        assert "</table>" in html
        assert "<tr>" in html

    def test_to_html_refused_row(self):
        report = EvalReport(suite_name="ref")
        report.results.append(
            QuestionResult(
                question="Q?",
                refusal_reason="Not in corpus",
                refusal_correct=True,
                expected_refusal=True,
            ),
        )
        report.compute_metrics()
        html = report.to_html()
        assert "refused" in html
        assert "Not in corpus" in html

    def test_to_html_error_row(self):
        report = EvalReport(suite_name="err")
        report.results.append(
            QuestionResult(question="Q?", error="boom"),
        )
        report.compute_metrics()
        html = report.to_html()
        assert "error" in html
        assert "boom" in html

    def test_to_html_escapes_special_chars(self):
        report = EvalReport(suite_name="<script>alert(1)</script>")
        report.compute_metrics()
        html = report.to_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# QuestionResult — data class
# ---------------------------------------------------------------------------


class TestQuestionResult:
    """Tests for QuestionResult defaults and immutability."""

    def test_defaults(self):
        r = QuestionResult(question="Q?")
        assert r.answer is None
        assert r.refusal_reason is None
        assert r.groundedness is None
        assert r.relevance is None
        assert r.refusal_correct is False
        assert r.expected_refusal is False
        assert r.latency_ms == 0.0
        assert r.error is None

    def test_frozen(self):
        r = QuestionResult(question="Q?")
        with pytest.raises(AttributeError):
            r.question = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AggregateMetrics — data class
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Tests for AggregateMetrics defaults and immutability."""

    def test_defaults(self):
        m = AggregateMetrics()
        assert m.total_questions == 0
        assert m.avg_groundedness == 0.0
        assert m.refusal_accuracy == 0.0

    def test_frozen(self):
        m = AggregateMetrics()
        with pytest.raises(AttributeError):
            m.total_questions = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify public API is importable from the eval package."""

    def test_eval_runner_exported(self):
        from src.eval import EvalRunner
        assert EvalRunner is not None

    def test_eval_report_exported(self):
        from src.eval import EvalReport
        assert EvalReport is not None

    def test_question_result_exported(self):
        from src.eval import QuestionResult
        assert QuestionResult is not None

    def test_aggregate_metrics_exported(self):
        from src.eval import AggregateMetrics
        assert AggregateMetrics is not None
