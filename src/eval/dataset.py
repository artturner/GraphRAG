"""Evaluation dataset definitions and loaders.

This module defines the data models for evaluation datasets and
provides loaders for JSON and YAML formats.  An evaluation dataset
is a collection of questions together with expectations about the
answers the RAG system should produce.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvalQuestion(BaseModel):
    """A single evaluation question with expected outcomes.

    Attributes:
        question: The natural-language question to ask.
        expected_answer_contains: Substrings that the generated answer
            **must** contain (case-insensitive check).  Empty list means
            no content check is applied.
        expected_citations_min: Minimum number of citations expected.
            ``0`` disables the check.
        expected_refusal: If ``True`` the system is expected to refuse
            (i.e. ``answer`` should be ``None`` and ``refusal_reason``
            should be set).
        refusal_reason: Optional human note explaining *why* a refusal
            is expected.  Not used for automated scoring — purely
            documentary.
        tags: Free-form tags for filtering / grouping questions.
        difficulty: Optional difficulty label (``easy``, ``medium``,
            ``hard``).

    Example:
        ```python
        q = EvalQuestion(
            question="What is federalism?",
            expected_answer_contains=["system of government"],
            expected_citations_min=1,
        )
        ```
    """

    question: str = Field(..., min_length=1, description="The question text")
    expected_answer_contains: list[str] = Field(
        default_factory=list,
        description="Substrings the answer must contain",
    )
    expected_citations_min: int = Field(
        default=0,
        ge=0,
        description="Minimum expected citations",
    )
    expected_refusal: bool = Field(
        default=False,
        description="Whether a refusal is expected",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Documentary note on why refusal is expected",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for filtering",
    )
    difficulty: str | None = Field(
        default=None,
        description="Difficulty label: easy, medium, hard",
    )

    @field_validator("difficulty")
    @classmethod
    def _validate_difficulty(cls, v: str | None) -> str | None:
        if v is not None and v not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be easy/medium/hard, got {v!r}")
        return v

    @model_validator(mode="after")
    def _refusal_consistency(self) -> "EvalQuestion":
        """When refusal is expected, answer-content checks make no sense."""
        if self.expected_refusal and self.expected_answer_contains:
            raise ValueError(
                "expected_answer_contains must be empty when "
                "expected_refusal is True"
            )
        return self


class EvalDataset(BaseModel):
    """A named collection of evaluation questions.

    Attributes:
        name: Short identifier for the dataset (e.g. ``"sample_qna"``).
        description: Optional human-readable description.
        questions: The list of :class:`EvalQuestion` entries.
        version: Optional version string for the dataset.

    Example:
        ```python
        ds = EvalDataset(
            name="sample",
            questions=[
                EvalQuestion(question="What is X?"),
            ],
        )
        ```
    """

    name: str = Field(..., min_length=1, description="Dataset identifier")
    description: str | None = Field(
        default=None,
        description="Human-readable description",
    )
    questions: list[EvalQuestion] = Field(
        ...,
        min_length=1,
        description="Evaluation questions",
    )
    version: str | None = Field(
        default=None,
        description="Dataset version string",
    )

    # -- convenience helpers -------------------------------------------

    def filter_by_tag(self, tag: str) -> list[EvalQuestion]:
        """Return questions that carry *tag*."""
        return [q for q in self.questions if tag in q.tags]

    def filter_by_difficulty(self, difficulty: str) -> list[EvalQuestion]:
        """Return questions with the given *difficulty* label."""
        return [q for q in self.questions if q.difficulty == difficulty]

    @property
    def refusal_questions(self) -> list[EvalQuestion]:
        """Return questions where a refusal is expected."""
        return [q for q in self.questions if q.expected_refusal]

    @property
    def answerable_questions(self) -> list[EvalQuestion]:
        """Return questions where an answer is expected."""
        return [q for q in self.questions if not q.expected_refusal]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> EvalDataset:
    """Load an :class:`EvalDataset` from a JSON or YAML file.

    The file extension determines the parser (``.json`` for JSON,
    ``.yaml`` / ``.yml`` for YAML).

    Args:
        path: Path to the dataset file.

    Returns:
        A validated :class:`EvalDataset`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is unsupported or the
            contents fail validation.
    """
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".json":
        data = _load_json(filepath)
    elif suffix in (".yaml", ".yml"):
        data = _load_yaml(filepath)
    else:
        raise ValueError(
            f"Unsupported dataset file extension: {suffix!r} "
            f"(expected .json, .yaml, or .yml)"
        )

    logger.info("Loaded evaluation dataset from %s", filepath)
    return EvalDataset.model_validate(data)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
