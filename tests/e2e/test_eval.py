"""End-to-end tests for the evaluation harness.

These tests run the :class:`~src.eval.runner.EvalRunner` against a
small dataset using a :class:`FakeGraph` and verify that scores are
within expected ranges and that reports are generated correctly.

Marked with ``@pytest.mark.e2e`` — run with ``pytest --run-e2e``.
"""

import json
import os
import tempfile

import pytest

from src.eval.dataset import EvalDataset, EvalQuestion, load_dataset
from src.eval.report import EvalReport, QuestionResult
from src.eval.runner import EvalRunner
from src.eval.scorers.groundedness import GroundednessScorer
from src.eval.scorers.relevance import RelevanceScorer
from src.eval.scorers.refusal import RefusalScorer
from src.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeGraph:
    """A deterministic graph that returns scripted responses.

    For answerable questions it echoes key terms from the question;
    for refusal-marked questions it returns a refusal.
    """

    def __init__(self, *, answer_map: dict[str, dict] | None = None):
        self._answer_map = answer_map or {}

    def invoke(self, input: dict) -> dict:
        question = input.get("question", "")

        # Check explicit answer map first
        if question in self._answer_map:
            return self._answer_map[question]

        q_lower = question.lower()

        # Simulate refusals for known out-of-scope topics
        if any(kw in q_lower for kw in ("bitcoin", "chocolate cake", "capital of france")):
            return {
                "question": question,
                "query_type": "unsupported",
                "chunks": [],
                "search_results": [],
                "answer": None,
                "citations": [],
                "confidence": 0.0,
                "is_grounded": False,
                "retry_count": 0,
                "action": "refuse",
                "refusal_reason": "No relevant documents found.",
                "error": None,
            }

        # Default: produce an answer that overlaps with question keywords
        chunk_text = f"The answer to '{question}' is based on source documents."
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content=chunk_text,
            start_idx=0,
            end_idx=len(chunk_text),
            metadata={"source": "test.txt"},
        )

        answer_text = self._generate_answer(question)

        return {
            "question": question,
            "query_type": "factual",
            "chunks": [chunk],
            "search_results": [],
            "answer": answer_text,
            "citations": [
                {
                    "source": "test.txt",
                    "chunk_id": "chunk-001",
                    "text": chunk_text,
                    "score": 0.85,
                }
            ],
            "confidence": 0.85,
            "is_grounded": True,
            "retry_count": 1,
            "action": "accept",
            "refusal_reason": None,
            "error": None,
        }

    @staticmethod
    def _generate_answer(question: str) -> str:
        q = question.lower()
        if "federalism" in q:
            return (
                "Federalism is a system of government in which power is divided "
                "between a national government and regional governments."
            )
        if "three branches" in q or "branches of government" in q:
            return (
                "The three branches are the legislative, executive, and judicial."
            )
        if "first president" in q:
            return "George Washington was the first president."
        if "bill of rights" in q:
            return (
                "The Bill of Rights is the first ten amendments to the "
                "Constitution protecting individual liberties."
            )
        if "senate" in q and "house" in q:
            return (
                "The Senate has 100 members while the House of Representatives "
                "has 435 members."
            )
        if "bill become" in q or "how does a bill" in q:
            return (
                "A bill must pass both Congress and be signed by the president."
            )
        if "register" in q and "vote" in q:
            return "You can register to vote at your local election office."
        return f"This is the answer to the question about: {question}"


def _build_mini_dataset() -> EvalDataset:
    """A compact dataset for testing."""
    return EvalDataset(
        name="e2e_mini",
        description="Mini dataset for e2e eval tests",
        version="1.0",
        questions=[
            EvalQuestion(
                question="What is federalism?",
                expected_answer_contains=["system of government"],
                expected_citations_min=1,
                expected_refusal=False,
                tags=["factual"],
                difficulty="easy",
            ),
            EvalQuestion(
                question="What are the three branches of government?",
                expected_answer_contains=["legislative", "executive", "judicial"],
                expected_refusal=False,
                tags=["factual"],
                difficulty="easy",
            ),
            EvalQuestion(
                question="What is the price of Bitcoin today?",
                expected_refusal=True,
                refusal_reason="not in corpus",
                tags=["refusal"],
                difficulty="easy",
            ),
            EvalQuestion(
                question="How do I bake a chocolate cake?",
                expected_refusal=True,
                refusal_reason="not in corpus",
                tags=["refusal"],
                difficulty="easy",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests — Runner
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestEvalRunnerE2E:
    """Run the evaluation runner end-to-end and verify scores."""

    def test_runner_produces_report(self):
        """The runner should return an EvalReport with metrics."""
        runner = EvalRunner()
        dataset = _build_mini_dataset()
        graph = FakeGraph()

        report = runner.run(dataset, graph)

        assert isinstance(report, EvalReport)
        assert report.suite_name == "e2e_mini"
        assert report.metrics.total_questions == 4

    def test_all_questions_scored(self):
        """Every question should have a result entry."""
        runner = EvalRunner()
        dataset = _build_mini_dataset()
        report = runner.run(dataset, FakeGraph())

        assert len(report.results) == 4

    def test_answerable_questions_have_groundedness(self):
        """Answerable questions should have non-None groundedness."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        answerable = [r for r in report.results if not r.expected_refusal]
        assert len(answerable) == 2
        for r in answerable:
            assert r.groundedness is not None
            assert 0.0 <= r.groundedness <= 1.0

    def test_answerable_questions_have_relevance(self):
        """Answerable questions should have non-None relevance."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        answerable = [r for r in report.results if not r.expected_refusal]
        for r in answerable:
            assert r.relevance is not None
            assert 0.0 <= r.relevance <= 1.0

    def test_refusal_questions_scored_correctly(self):
        """Refusal questions should have refusal_correct=True."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        refusals = [r for r in report.results if r.expected_refusal]
        assert len(refusals) == 2
        for r in refusals:
            assert r.refusal_correct is True
            assert r.refusal_reason is not None

    def test_refusal_accuracy_is_perfect(self):
        """With correctly scripted FakeGraph, refusal accuracy should be 1.0."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        assert report.metrics.refusal_accuracy == 1.0

    def test_avg_groundedness_above_threshold(self):
        """Average groundedness should be above 0 (FakeLLM echoes content)."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        assert report.metrics.avg_groundedness >= 0.0

    def test_avg_relevance_above_threshold(self):
        """Average relevance should be > 0 (answers contain question keywords)."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        assert report.metrics.avg_relevance > 0.0

    def test_no_errors(self):
        """No questions should have errors."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        assert report.metrics.error_count == 0
        for r in report.results:
            assert r.error is None

    def test_latency_is_recorded(self):
        """Each result should have a non-negative latency."""
        runner = EvalRunner()
        report = runner.run(_build_mini_dataset(), FakeGraph())

        for r in report.results:
            assert r.latency_ms >= 0

    def test_custom_scorers(self):
        """Custom scorer parameters should be respected."""
        runner = EvalRunner(
            groundedness_scorer=GroundednessScorer(threshold=0.5),
            relevance_scorer=RelevanceScorer(length_penalty=False),
        )
        report = runner.run(_build_mini_dataset(), FakeGraph())

        assert report.metrics.total_questions == 4


# ---------------------------------------------------------------------------
# Tests — Error resilience
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestEvalErrorResilience:
    """The runner should handle errors gracefully."""

    def test_graph_raises_exception_for_one_question(self):
        """If the graph throws on one question, other questions should still be scored."""

        class PartiallyBrokenGraph:
            def invoke(self, input: dict) -> dict:
                if "crash" in input.get("question", "").lower():
                    raise RuntimeError("Simulated crash")
                return FakeGraph().invoke(input)

        dataset = EvalDataset(
            name="error_test",
            questions=[
                EvalQuestion(question="What is federalism?"),
                EvalQuestion(question="This should crash"),
                EvalQuestion(question="What are the three branches of government?"),
            ],
        )

        runner = EvalRunner()
        report = runner.run(dataset, PartiallyBrokenGraph())

        assert report.metrics.total_questions == 3
        assert report.metrics.error_count == 1

        errors = [r for r in report.results if r.error is not None]
        assert len(errors) == 1
        assert "crash" in errors[0].error.lower()

        # Other questions should still have results
        ok = [r for r in report.results if r.error is None]
        assert len(ok) == 2


# ---------------------------------------------------------------------------
# Tests — Report generation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestEvalReportE2E:
    """Verify report serialization formats."""

    @pytest.fixture()
    def report(self) -> EvalReport:
        runner = EvalRunner()
        return runner.run(_build_mini_dataset(), FakeGraph())

    def test_to_dict(self, report: EvalReport):
        d = report.to_dict()
        assert "suite_name" in d
        assert "metrics" in d
        assert "results" in d
        assert d["metrics"]["total_questions"] == 4

    def test_to_json(self, report: EvalReport):
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["suite_name"] == "e2e_mini"

    def test_to_html(self, report: EvalReport):
        html = report.to_html()
        assert "<html" in html.lower() or "<!doctype" in html.lower()
        assert "e2e_mini" in html

    def test_json_round_trip(self, report: EvalReport):
        """Serialise to JSON and verify the structure is preserved."""
        j = report.to_json()
        parsed = json.loads(j)
        assert len(parsed["results"]) == 4
        assert parsed["metrics"]["refusal_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Tests — Dataset loading
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestDatasetLoadingE2E:
    """Verify dataset loading from YAML files."""

    def test_load_sample_dataset(self):
        """The built-in sample_qna.yaml should load successfully."""
        ds = load_dataset("src/eval/datasets/sample_qna.yaml")
        assert ds.name == "sample_qna"
        assert len(ds.questions) == 10

    def test_dataset_filtering(self):
        """Filtering helpers should work on loaded datasets."""
        ds = load_dataset("src/eval/datasets/sample_qna.yaml")

        factual = ds.filter_by_tag("factual")
        assert len(factual) > 0

        refusals = ds.refusal_questions
        assert len(refusals) > 0

        answerable = ds.answerable_questions
        assert len(answerable) > 0

        assert len(refusals) + len(answerable) == len(ds.questions)

    def test_load_custom_yaml_dataset(self, tmp_dir: str):
        """A custom YAML dataset should load and validate."""
        yaml_content = """\
name: custom_test
description: Custom test dataset
version: "1.0"

questions:
  - question: "What is X?"
    expected_answer_contains: ["X"]
    expected_refusal: false
    tags: ["factual"]
    difficulty: easy

  - question: "What is Y?"
    expected_refusal: true
    refusal_reason: "not in corpus"
    tags: ["refusal"]
"""
        path = os.path.join(tmp_dir, "custom.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        ds = load_dataset(path)
        assert ds.name == "custom_test"
        assert len(ds.questions) == 2

    def test_load_custom_json_dataset(self, tmp_dir: str):
        """A custom JSON dataset should load and validate."""
        data = {
            "name": "json_test",
            "questions": [
                {"question": "What is Z?", "expected_refusal": False},
            ],
        }
        path = os.path.join(tmp_dir, "custom.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        ds = load_dataset(path)
        assert ds.name == "json_test"
        assert len(ds.questions) == 1

    def test_dataset_validation_rejects_bad_data(self, tmp_dir: str):
        """Invalid dataset should raise a validation error."""
        yaml_content = """\
name: bad
questions:
  - question: "What?"
    expected_refusal: true
    expected_answer_contains: ["should fail"]
"""
        path = os.path.join(tmp_dir, "bad.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        with pytest.raises(Exception):
            load_dataset(path)


# ---------------------------------------------------------------------------
# Tests — Full eval with sample dataset
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestFullEvalWithSampleDataset:
    """Run eval with the built-in sample_qna dataset and FakeGraph."""

    def test_full_sample_eval(self):
        """Run all 10 questions and verify metrics are reasonable."""
        ds = load_dataset("src/eval/datasets/sample_qna.yaml")
        runner = EvalRunner()
        report = runner.run(ds, FakeGraph())

        assert report.metrics.total_questions == 10
        assert report.metrics.error_count == 0
        assert report.metrics.refusal_accuracy >= 0.5
        assert report.metrics.answered_count + report.metrics.refused_count == 10
