"""Relevance scorer for RAG evaluation.

This scorer measures how well an answer addresses the original question.
It uses token overlap between the question and answer, weighted by term
frequency so that rare (more informative) terms contribute more.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from and or but not no nor so yet as if it its this that "
    "these those i me my we our you your he him his she her they them "
    "their what which who whom how when where why".split()
)

# Question-specific words that shouldn't count toward relevance
_QUESTION_FILLER: frozenset[str] = frozenset(
    "please tell explain describe define about".split()
)


def _tokenize(text: str) -> list[str]:
    """Return lowercased non-stop-word tokens (preserving duplicates)."""
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in _STOP_WORDS and tok not in _QUESTION_FILLER and len(tok) > 1
    ]


def _unique_tokens(text: str) -> set[str]:
    """Return the *set* of meaningful tokens in *text*."""
    return set(_tokenize(text))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelevanceResult:
    """Full output of the relevance scorer.

    Attributes:
        score: Overall relevance score in [0, 1].
        question_tokens: Meaningful tokens extracted from the question.
        matched_tokens: Question tokens that appear in the answer.
        unmatched_tokens: Question tokens absent from the answer.
    """

    score: float
    question_tokens: list[str] = field(default_factory=list)
    matched_tokens: list[str] = field(default_factory=list)
    unmatched_tokens: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class RelevanceScorer:
    """Score how relevant an answer is to the original question.

    The scorer extracts meaningful tokens from the question, checks
    which of those tokens appear in the answer, and computes a
    weighted recall score.  Tokens that are rarer in the answer
    (measured by inverse frequency) receive a higher weight so that
    topical keywords matter more than common terms.

    An optional *length_penalty* reduces the score when the answer is
    very short relative to the question — a single-word answer is
    unlikely to be a complete response.

    Args:
        length_penalty: If ``True`` (default), apply a mild penalty
            when the answer contains fewer meaningful tokens than the
            question.

    Example:
        ```python
        scorer = RelevanceScorer()
        score = scorer.score("Federalism is a system of government.",
                             "What is federalism?")
        ```
    """

    def __init__(self, *, length_penalty: bool = True) -> None:
        self.length_penalty = length_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, answer: str, question: str) -> float:
        """Return a single relevance score in [0, 1].

        Args:
            answer: The generated answer text.
            question: The original question.

        Returns:
            ``1.0`` when every question keyword appears in the answer,
            ``0.0`` when none do (or inputs are empty).
        """
        return self.score_detailed(answer, question).score

    def score_detailed(
        self,
        answer: str,
        question: str,
    ) -> RelevanceResult:
        """Return a detailed relevance result.

        Args:
            answer: The generated answer text.
            question: The original question.

        Returns:
            A :class:`RelevanceResult` with the overall score and
            token-level detail.
        """
        if not question or not question.strip():
            return RelevanceResult(score=0.0)

        if not answer or not answer.strip():
            return RelevanceResult(score=0.0)

        q_tokens = list(dict.fromkeys(_tokenize(question)))  # unique, ordered
        if not q_tokens:
            # Question contained only stop-words — treat as fully relevant
            return RelevanceResult(score=1.0)

        answer_token_set = _unique_tokens(answer)
        answer_token_counts = Counter(_tokenize(answer))
        total_answer_tokens = sum(answer_token_counts.values()) or 1

        matched: list[str] = []
        unmatched: list[str] = []
        weighted_hit = 0.0
        total_weight = 0.0

        for tok in q_tokens:
            # IDF-like weight: rarer tokens in the answer are more
            # informative when they *do* appear.
            freq = answer_token_counts.get(tok, 0) / total_answer_tokens
            weight = 1.0 + (1.0 - freq)  # range [1.0, 2.0]

            total_weight += weight

            if tok in answer_token_set:
                matched.append(tok)
                weighted_hit += weight
            else:
                unmatched.append(tok)

        raw_score = weighted_hit / total_weight if total_weight else 0.0

        # Optional length penalty
        if self.length_penalty and len(answer_token_set) < len(q_tokens):
            ratio = len(answer_token_set) / len(q_tokens)
            penalty = 0.5 + 0.5 * ratio  # range [0.5, 1.0]
            raw_score *= penalty

        final = round(min(max(raw_score, 0.0), 1.0), 4)

        return RelevanceResult(
            score=final,
            question_tokens=q_tokens,
            matched_tokens=matched,
            unmatched_tokens=unmatched,
        )
