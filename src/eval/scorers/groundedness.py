"""Groundedness scorer for RAG evaluation.

This scorer measures how well an answer is supported by the retrieved
chunks.  It builds on the token-overlap engine in
:mod:`src.graphs.grounding` but exposes an evaluation-oriented API
with per-sentence detail and a single ``0.0–1.0`` score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.types import Chunk


# ---------------------------------------------------------------------------
# Helpers (shared with src.graphs.grounding but kept local to avoid
# coupling the eval package to the graph package)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_STOP_WORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from and or but not no nor so yet as if it its this that "
    "these those i me my we our you your he him his she her they them "
    "their what which who whom how when where why".split()
)


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercased non-stop-word tokens."""
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in _STOP_WORDS and len(tok) > 1
    }


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, discarding blanks."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceScore:
    """Groundedness verdict for a single sentence.

    Attributes:
        sentence: The original sentence text.
        score: Overlap ratio in [0, 1].
        supported: Whether the sentence meets the support threshold.
    """

    sentence: str
    score: float
    supported: bool


@dataclass(frozen=True)
class GroundednessResult:
    """Full output of the groundedness scorer.

    Attributes:
        score: Overall groundedness score in [0, 1].
        sentence_scores: Per-sentence breakdown.
        supported_count: Number of supported sentences.
        unsupported_count: Number of unsupported sentences.
    """

    score: float
    sentence_scores: list[SentenceScore] = field(default_factory=list)
    supported_count: int = 0
    unsupported_count: int = 0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class GroundednessScorer:
    """Score how well an answer is grounded in retrieved chunks.

    The scorer splits the answer into sentences, computes the token
    overlap between each sentence and the combined chunk content, and
    returns an overall score equal to the fraction of sentences that
    exceed the *threshold*.

    Args:
        threshold: Minimum token-overlap ratio for a sentence to be
            considered supported.  Defaults to ``0.3``.

    Example:
        ```python
        scorer = GroundednessScorer()
        result = scorer.score_detailed(
            "Federalism divides power between governments.",
            [Chunk(content="Federalism divides power ...")],
        )
        print(result.score)  # 1.0
        ```
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, answer: str, chunks: Sequence[Chunk]) -> float:
        """Return a single groundedness score in [0, 1].

        Args:
            answer: The generated answer text.
            chunks: Retrieved chunks that should support the answer.

        Returns:
            ``1.0`` when every sentence is supported, ``0.0`` when none
            are (or when the answer / chunks are empty).
        """
        return self.score_detailed(answer, chunks).score

    def score_detailed(
        self,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> GroundednessResult:
        """Return a detailed groundedness result with per-sentence scores.

        Args:
            answer: The generated answer text.
            chunks: Retrieved chunks that should support the answer.

        Returns:
            A :class:`GroundednessResult` containing the overall score
            and per-sentence breakdowns.
        """
        if not answer or not answer.strip():
            return GroundednessResult(score=0.0)

        if not chunks:
            sentences = _split_sentences(answer)
            sentence_scores = [
                SentenceScore(sentence=s, score=0.0, supported=False)
                for s in sentences
            ]
            return GroundednessResult(
                score=0.0,
                sentence_scores=sentence_scores,
                supported_count=0,
                unsupported_count=len(sentences),
            )

        # Build combined context tokens
        context_tokens: set[str] = set()
        for chunk in chunks:
            context_tokens |= _tokenize(chunk.content)

        sentences = _split_sentences(answer)
        if not sentences:
            return GroundednessResult(score=0.0)

        sentence_scores: list[SentenceScore] = []
        supported = 0
        unsupported = 0

        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)

            if not sentence_tokens:
                # Pure stop-words / punctuation — treat as supported
                sentence_scores.append(
                    SentenceScore(sentence=sentence, score=1.0, supported=True)
                )
                supported += 1
                continue

            overlap = len(sentence_tokens & context_tokens)
            ratio = round(overlap / len(sentence_tokens), 4)
            is_supported = ratio >= self.threshold

            sentence_scores.append(
                SentenceScore(
                    sentence=sentence,
                    score=ratio,
                    supported=is_supported,
                )
            )

            if is_supported:
                supported += 1
            else:
                unsupported += 1

        overall = round(supported / len(sentences), 4) if sentences else 0.0

        return GroundednessResult(
            score=overall,
            sentence_scores=sentence_scores,
            supported_count=supported,
            unsupported_count=unsupported,
        )
