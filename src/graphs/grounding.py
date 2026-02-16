"""Grounding verification for RAG answers.

This module provides a :class:`GroundingChecker` that determines whether
an answer is supported by the retrieved chunks.  The current implementation
uses simple token-overlap heuristics; it can be replaced with an
LLM-based grounding model in the future.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.types import Chunk


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundingResult:
    """Outcome of a grounding check.

    Attributes:
        is_grounded: ``True`` if the answer is sufficiently supported.
        confidence: A score in [0, 1] representing how well the answer
            is supported by the context.
        supported_claims: Sentences from the answer that *are* backed
            by the chunks.
        unsupported_claims: Sentences from the answer that could *not*
            be matched to any chunk.
    """

    is_grounded: bool
    confidence: float
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Common stop words excluded from overlap scoring
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
# Grounding checker
# ---------------------------------------------------------------------------

class GroundingChecker:
    """Check whether an answer is grounded in a set of chunks.

    The checker splits the answer into sentences, tokenises each sentence
    and every chunk, and measures the token overlap.  A sentence is
    considered *supported* when its overlap ratio exceeds ``threshold``.

    Args:
        threshold: Minimum token-overlap ratio for a sentence to be
            considered supported.  Defaults to ``0.3``.
        min_confidence: Overall confidence threshold below which the
            answer is considered *not grounded*.  Defaults to ``0.5``.

    Example:
        ```python
        checker = GroundingChecker()
        result = checker.check_grounding(
            "Federalism divides power between governments.",
            [Chunk(content="Federalism divides power between ...")],
        )
        assert result.is_grounded
        ```
    """

    def __init__(
        self,
        threshold: float = 0.3,
        min_confidence: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.min_confidence = min_confidence

    def check_grounding(
        self,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> GroundingResult:
        """Run the grounding check.

        Args:
            answer: The generated answer text.
            chunks: Retrieved chunks that should support the answer.

        Returns:
            A :class:`GroundingResult` with per-sentence verdicts and
            an overall confidence score.
        """
        if not answer or not answer.strip():
            return GroundingResult(
                is_grounded=False,
                confidence=0.0,
                unsupported_claims=["(empty answer)"],
            )

        if not chunks:
            sentences = _split_sentences(answer)
            return GroundingResult(
                is_grounded=False,
                confidence=0.0,
                unsupported_claims=sentences or ["(empty answer)"],
            )

        # Build a single token set from all chunk content
        context_tokens: set[str] = set()
        for chunk in chunks:
            context_tokens |= _tokenize(chunk.content)

        sentences = _split_sentences(answer)
        if not sentences:
            return GroundingResult(is_grounded=False, confidence=0.0)

        supported: list[str] = []
        unsupported: list[str] = []

        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                # Purely stop-words or punctuation — treat as supported
                supported.append(sentence)
                continue

            overlap = len(sentence_tokens & context_tokens)
            ratio = overlap / len(sentence_tokens)

            if ratio >= self.threshold:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        confidence = len(supported) / len(sentences) if sentences else 0.0
        is_grounded = confidence >= self.min_confidence

        return GroundingResult(
            is_grounded=is_grounded,
            confidence=round(confidence, 4),
            supported_claims=supported,
            unsupported_claims=unsupported,
        )
