"""Tests for the groundedness scorer."""

from __future__ import annotations

import pytest

from src.eval.scorers import GroundednessResult, GroundednessScorer, SentenceScore
from src.types import Chunk


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


# ---------------------------------------------------------------------------
# score() — fully grounded
# ---------------------------------------------------------------------------

class TestGroundedAnswer:
    """Tests where the answer is fully supported by chunks."""

    def test_exact_match(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Federalism divides power between governments.")]
        score = scorer.score(
            "Federalism divides power between governments.", chunks
        )
        assert score == 1.0

    def test_high_overlap(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Machine learning uses statistical models to learn from data.")]
        score = scorer.score(
            "Machine learning uses statistical models.", chunks
        )
        assert score >= 0.9

    def test_multiple_chunks(self):
        scorer = GroundednessScorer()
        chunks = [
            _chunk("Federalism is a system of government."),
            _chunk("Democracy allows citizens to vote in elections."),
        ]
        score = scorer.score(
            "Federalism is a system of government. "
            "Democracy allows citizens to vote in elections.",
            chunks,
        )
        assert score == 1.0

    def test_single_sentence_supported(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Python is a programming language.")]
        score = scorer.score("Python is a programming language.", chunks)
        assert score == 1.0


# ---------------------------------------------------------------------------
# score() — fully ungrounded
# ---------------------------------------------------------------------------

class TestUngroundedAnswer:
    """Tests where the answer is NOT supported by chunks."""

    def test_completely_unrelated(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Photosynthesis converts sunlight into chemical energy.")]
        score = scorer.score(
            "Quantum entanglement connects distant particles instantly.",
            chunks,
        )
        assert score == 0.0

    def test_no_chunks(self):
        scorer = GroundednessScorer()
        score = scorer.score("Some answer.", [])
        assert score == 0.0

    def test_empty_answer(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("content")]
        score = scorer.score("", chunks)
        assert score == 0.0

    def test_whitespace_answer(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("content")]
        score = scorer.score("   ", chunks)
        assert score == 0.0

    def test_no_chunks_and_no_answer(self):
        scorer = GroundednessScorer()
        score = scorer.score("", [])
        assert score == 0.0


# ---------------------------------------------------------------------------
# score() — partial grounding
# ---------------------------------------------------------------------------

class TestPartialGrounding:
    """Tests where only some sentences are supported."""

    def test_one_supported_one_not(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Federalism divides governmental power.")]
        score = scorer.score(
            "Federalism divides governmental power. "
            "Aliens invaded the planet yesterday.",
            chunks,
        )
        assert 0.0 < score < 1.0
        assert score == pytest.approx(0.5, abs=0.01)

    def test_two_supported_one_not(self):
        scorer = GroundednessScorer()
        chunks = [
            _chunk("Federalism divides power."),
            _chunk("Democracy enables voting."),
        ]
        score = scorer.score(
            "Federalism divides power. "
            "Democracy enables voting. "
            "Quantum computing revolutionizes cryptography.",
            chunks,
        )
        # 2 out of 3 supported
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_score_between_0_and_1(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Machine learning algorithms learn from data.")]
        score = scorer.score(
            "Machine learning algorithms learn from data. "
            "Blockchain decentralizes transactions.",
            chunks,
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_detailed() — result structure
# ---------------------------------------------------------------------------

class TestScoreDetailed:
    """Tests for the detailed scoring result."""

    def test_returns_groundedness_result(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Content here.")]
        result = scorer.score_detailed("Content here.", chunks)
        assert isinstance(result, GroundednessResult)

    def test_sentence_scores_populated(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Federalism divides power.")]
        result = scorer.score_detailed(
            "Federalism divides power. Aliens exist.",
            chunks,
        )
        assert len(result.sentence_scores) == 2
        assert all(isinstance(s, SentenceScore) for s in result.sentence_scores)

    def test_supported_and_unsupported_counts(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Federalism divides power.")]
        result = scorer.score_detailed(
            "Federalism divides power. Aliens exist.",
            chunks,
        )
        assert result.supported_count == 1
        assert result.unsupported_count == 1

    def test_sentence_score_fields(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Python programming language features duck typing.")]
        result = scorer.score_detailed(
            "Python programming language features duck typing.",
            chunks,
        )
        ss = result.sentence_scores[0]
        assert ss.sentence == "Python programming language features duck typing."
        assert ss.score >= 0.3
        assert ss.supported is True

    def test_empty_answer_detailed(self):
        scorer = GroundednessScorer()
        result = scorer.score_detailed("", [_chunk("content")])
        assert result.score == 0.0
        assert result.sentence_scores == []
        assert result.supported_count == 0
        assert result.unsupported_count == 0

    def test_no_chunks_detailed(self):
        scorer = GroundednessScorer()
        result = scorer.score_detailed("Some answer.", [])
        assert result.score == 0.0
        assert len(result.sentence_scores) == 1
        assert result.sentence_scores[0].supported is False
        assert result.unsupported_count == 1

    def test_overall_score_matches_score_method(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Federalism divides power.")]
        answer = "Federalism divides power. Aliens exist."
        assert scorer.score(answer, chunks) == scorer.score_detailed(answer, chunks).score


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

class TestCustomThreshold:
    """Tests for custom threshold parameter."""

    def test_low_threshold_more_lenient(self):
        scorer = GroundednessScorer(threshold=0.1)
        chunks = [_chunk("Federalism.")]
        # With a very low threshold, even marginal overlap succeeds
        score = scorer.score(
            "Federalism divides power between national and state governments.",
            chunks,
        )
        assert score >= 0.5

    def test_high_threshold_more_strict(self):
        scorer = GroundednessScorer(threshold=0.99)
        chunks = [_chunk("Federalism.")]
        score = scorer.score(
            "Federalism divides power between national and state governments.",
            chunks,
        )
        assert score < 1.0

    def test_zero_threshold_everything_supported(self):
        scorer = GroundednessScorer(threshold=0.0)
        chunks = [_chunk("anything")]
        score = scorer.score(
            "Completely unrelated content about quantum physics.",
            chunks,
        )
        # threshold=0.0 means any non-empty overlap ratio >= 0.0
        assert score == 1.0

    def test_one_threshold_nothing_supported(self):
        scorer = GroundednessScorer(threshold=1.0)
        chunks = [_chunk("Partial match only here.")]
        score = scorer.score(
            "Partial match only here plus some extra words.",
            chunks,
        )
        # Would need 100% overlap, unlikely with extra words
        assert score < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case tests."""

    def test_stopword_only_sentence(self):
        """A sentence of only stop-words should be treated as supported."""
        scorer = GroundednessScorer()
        chunks = [_chunk("Some content.")]
        # "It is." contains only stop-words/short tokens
        result = scorer.score_detailed("It is. Some content.", chunks)
        # "It is." → no meaningful tokens → supported
        assert result.supported_count >= 1

    def test_single_word_answer(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("Federalism.")]
        score = scorer.score("Federalism.", chunks)
        assert score == 1.0

    def test_numbers_in_answer(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("The Constitution has 27 amendments.")]
        score = scorer.score("The Constitution has 27 amendments.", chunks)
        assert score == 1.0

    def test_case_insensitive(self):
        scorer = GroundednessScorer()
        chunks = [_chunk("FEDERALISM DIVIDES POWER.")]
        score = scorer.score("federalism divides power.", chunks)
        assert score == 1.0

    def test_many_chunks(self):
        scorer = GroundednessScorer()
        chunks = [_chunk(f"Topic {i} content here.", f"c-{i}") for i in range(20)]
        score = scorer.score("Topic 5 content here.", chunks)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Data class immutability
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Tests for result dataclass properties."""

    def test_sentence_score_frozen(self):
        ss = SentenceScore(sentence="test", score=0.5, supported=True)
        with pytest.raises(AttributeError):
            ss.score = 0.9  # type: ignore[misc]

    def test_groundedness_result_frozen(self):
        r = GroundednessResult(score=0.5)
        with pytest.raises(AttributeError):
            r.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify scorer is importable from the scorers package."""

    def test_scorer_exported(self):
        from src.eval.scorers import GroundednessScorer as cls
        assert cls is not None

    def test_result_exported(self):
        from src.eval.scorers import GroundednessResult
        assert GroundednessResult is not None

    def test_sentence_score_exported(self):
        from src.eval.scorers import SentenceScore
        assert SentenceScore is not None
