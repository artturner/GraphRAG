"""Tests for the evaluation dataset models and loaders."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from src.eval import EvalDataset, EvalQuestion, load_dataset


# ---------------------------------------------------------------------------
# EvalQuestion — construction
# ---------------------------------------------------------------------------

class TestEvalQuestionCreation:
    """Tests for EvalQuestion model creation."""

    def test_minimal(self):
        q = EvalQuestion(question="What is X?")
        assert q.question == "What is X?"
        assert q.expected_answer_contains == []
        assert q.expected_citations_min == 0
        assert q.expected_refusal is False
        assert q.refusal_reason is None
        assert q.tags == []
        assert q.difficulty is None

    def test_full(self):
        q = EvalQuestion(
            question="What is federalism?",
            expected_answer_contains=["system", "power"],
            expected_citations_min=2,
            expected_refusal=False,
            tags=["factual"],
            difficulty="medium",
        )
        assert q.expected_answer_contains == ["system", "power"]
        assert q.expected_citations_min == 2
        assert q.tags == ["factual"]
        assert q.difficulty == "medium"

    def test_refusal_question(self):
        q = EvalQuestion(
            question="What is the capital of Mars?",
            expected_refusal=True,
            refusal_reason="not in corpus",
        )
        assert q.expected_refusal is True
        assert q.refusal_reason == "not in corpus"


# ---------------------------------------------------------------------------
# EvalQuestion — validation
# ---------------------------------------------------------------------------

class TestEvalQuestionValidation:
    """Tests for EvalQuestion field validation."""

    def test_empty_question_rejected(self):
        with pytest.raises(ValueError):
            EvalQuestion(question="")

    def test_negative_citations_min_rejected(self):
        with pytest.raises(ValueError):
            EvalQuestion(question="Q?", expected_citations_min=-1)

    def test_invalid_difficulty_rejected(self):
        with pytest.raises(ValueError, match="difficulty"):
            EvalQuestion(question="Q?", difficulty="extreme")

    def test_valid_difficulties(self):
        for d in ("easy", "medium", "hard"):
            q = EvalQuestion(question="Q?", difficulty=d)
            assert q.difficulty == d

    def test_refusal_with_answer_contains_rejected(self):
        with pytest.raises(ValueError, match="expected_answer_contains"):
            EvalQuestion(
                question="Q?",
                expected_refusal=True,
                expected_answer_contains=["something"],
            )

    def test_refusal_with_empty_answer_contains_ok(self):
        q = EvalQuestion(question="Q?", expected_refusal=True)
        assert q.expected_answer_contains == []


# ---------------------------------------------------------------------------
# EvalDataset — construction
# ---------------------------------------------------------------------------

class TestEvalDatasetCreation:
    """Tests for EvalDataset model creation."""

    def test_minimal(self):
        ds = EvalDataset(
            name="test",
            questions=[EvalQuestion(question="Q?")],
        )
        assert ds.name == "test"
        assert len(ds.questions) == 1
        assert ds.description is None
        assert ds.version is None

    def test_full(self):
        ds = EvalDataset(
            name="full",
            description="A full dataset",
            version="2.0",
            questions=[
                EvalQuestion(question="Q1?"),
                EvalQuestion(question="Q2?"),
            ],
        )
        assert ds.description == "A full dataset"
        assert ds.version == "2.0"
        assert len(ds.questions) == 2


# ---------------------------------------------------------------------------
# EvalDataset — validation
# ---------------------------------------------------------------------------

class TestEvalDatasetValidation:
    """Tests for EvalDataset field validation."""

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            EvalDataset(name="", questions=[EvalQuestion(question="Q?")])

    def test_empty_questions_rejected(self):
        with pytest.raises(ValueError):
            EvalDataset(name="test", questions=[])

    def test_invalid_question_in_list_rejected(self):
        with pytest.raises(ValueError):
            EvalDataset(
                name="test",
                questions=[{"question": ""}],  # type: ignore[list-item]
            )


# ---------------------------------------------------------------------------
# EvalDataset — helpers
# ---------------------------------------------------------------------------

class TestEvalDatasetHelpers:
    """Tests for convenience filtering methods."""

    def _dataset(self) -> EvalDataset:
        return EvalDataset(
            name="helpers",
            questions=[
                EvalQuestion(
                    question="Q1?",
                    tags=["factual", "gov"],
                    difficulty="easy",
                ),
                EvalQuestion(
                    question="Q2?",
                    expected_refusal=True,
                    tags=["refusal"],
                    difficulty="hard",
                ),
                EvalQuestion(
                    question="Q3?",
                    tags=["factual"],
                    difficulty="easy",
                ),
            ],
        )

    def test_filter_by_tag(self):
        ds = self._dataset()
        assert len(ds.filter_by_tag("factual")) == 2
        assert len(ds.filter_by_tag("refusal")) == 1
        assert len(ds.filter_by_tag("nonexistent")) == 0

    def test_filter_by_difficulty(self):
        ds = self._dataset()
        assert len(ds.filter_by_difficulty("easy")) == 2
        assert len(ds.filter_by_difficulty("hard")) == 1
        assert len(ds.filter_by_difficulty("medium")) == 0

    def test_refusal_questions(self):
        ds = self._dataset()
        refusals = ds.refusal_questions
        assert len(refusals) == 1
        assert refusals[0].question == "Q2?"

    def test_answerable_questions(self):
        ds = self._dataset()
        answerable = ds.answerable_questions
        assert len(answerable) == 2


# ---------------------------------------------------------------------------
# load_dataset — YAML
# ---------------------------------------------------------------------------

class TestLoadDatasetYAML:
    """Tests for loading datasets from YAML files."""

    def test_load_yaml(self, tmp_path: Path):
        data = {
            "name": "yaml_test",
            "questions": [
                {"question": "What is X?", "expected_refusal": False},
                {"question": "What is Y?", "expected_refusal": True},
            ],
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")

        ds = load_dataset(path)

        assert ds.name == "yaml_test"
        assert len(ds.questions) == 2

    def test_load_yml_extension(self, tmp_path: Path):
        data = {
            "name": "yml_test",
            "questions": [{"question": "Q?"}],
        }
        path = tmp_path / "test.yml"
        path.write_text(yaml.dump(data), encoding="utf-8")

        ds = load_dataset(path)

        assert ds.name == "yml_test"

    def test_load_sample_dataset(self):
        """Load the bundled sample_qna.yaml and validate it."""
        sample = Path(__file__).resolve().parents[2] / "src" / "eval" / "datasets" / "sample_qna.yaml"
        ds = load_dataset(sample)

        assert ds.name == "sample_qna"
        assert len(ds.questions) == 10
        assert ds.version == "1.0"
        assert len(ds.refusal_questions) == 3
        assert len(ds.answerable_questions) == 7

    def test_sample_dataset_tags(self):
        sample = Path(__file__).resolve().parents[2] / "src" / "eval" / "datasets" / "sample_qna.yaml"
        ds = load_dataset(sample)

        assert len(ds.filter_by_tag("factual")) >= 1
        assert len(ds.filter_by_tag("procedural")) >= 1
        assert len(ds.filter_by_tag("refusal")) >= 1

    def test_sample_dataset_difficulties(self):
        sample = Path(__file__).resolve().parents[2] / "src" / "eval" / "datasets" / "sample_qna.yaml"
        ds = load_dataset(sample)

        assert len(ds.filter_by_difficulty("easy")) >= 1
        assert len(ds.filter_by_difficulty("medium")) >= 1


# ---------------------------------------------------------------------------
# load_dataset — JSON
# ---------------------------------------------------------------------------

class TestLoadDatasetJSON:
    """Tests for loading datasets from JSON files."""

    def test_load_json(self, tmp_path: Path):
        data = {
            "name": "json_test",
            "questions": [
                {"question": "What is X?"},
                {"question": "What is Y?", "expected_refusal": True},
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        ds = load_dataset(path)

        assert ds.name == "json_test"
        assert len(ds.questions) == 2

    def test_json_full_fields(self, tmp_path: Path):
        data = {
            "name": "full",
            "description": "Full JSON dataset",
            "version": "1.0",
            "questions": [
                {
                    "question": "What is federalism?",
                    "expected_answer_contains": ["system"],
                    "expected_citations_min": 1,
                    "expected_refusal": False,
                    "tags": ["factual"],
                    "difficulty": "easy",
                },
            ],
        }
        path = tmp_path / "full.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        ds = load_dataset(path)

        assert ds.description == "Full JSON dataset"
        q = ds.questions[0]
        assert q.expected_answer_contains == ["system"]
        assert q.expected_citations_min == 1
        assert q.tags == ["factual"]
        assert q.difficulty == "easy"


# ---------------------------------------------------------------------------
# load_dataset — errors
# ---------------------------------------------------------------------------

class TestLoadDatasetErrors:
    """Tests for loader error handling."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/dataset.yaml")

    def test_unsupported_extension(self, tmp_path: Path):
        path = tmp_path / "data.csv"
        path.write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(path)

    def test_invalid_yaml_content(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.dump({"name": "bad", "questions": []}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            load_dataset(path)

    def test_invalid_json_content(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"name": "bad", "questions": [{"question": ""}]}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            load_dataset(path)

    def test_string_path_accepted(self, tmp_path: Path):
        data = {"name": "str_path", "questions": [{"question": "Q?"}]}
        path = tmp_path / "str.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")

        ds = load_dataset(str(path))

        assert ds.name == "str_path"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify public API is importable from the eval package."""

    def test_eval_question_exported(self):
        from src.eval import EvalQuestion
        assert EvalQuestion is not None

    def test_eval_dataset_exported(self):
        from src.eval import EvalDataset
        assert EvalDataset is not None

    def test_load_dataset_exported(self):
        from src.eval import load_dataset
        assert callable(load_dataset)
