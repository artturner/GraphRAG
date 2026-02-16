"""Tests for the relevance scorer."""

from __future__ import annotations

import pytest

from src.eval.scorers import RelevanceResult, RelevanceScorer


# ---------------------------------------------------------------------------
# score() — fully relevant
# ---------------------------------------------------------------------------

class TestRelevantAnswer:
    """Tests where the answer is fully relevant to the question."""

    def test_exact_keyword_match(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Federalism is a system of government that divides power.",
            "What is federalism?",
        )
        assert score == 1.0

    def test_all_question_tokens_present(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Machine learning uses statistical models to learn patterns from data.",
            "How does machine learning use statistical models?",
        )
        assert score >= 0.7

    def test_verbose_answer_covers_question(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Python is a high-level programming language known for its "
            "readability. It supports multiple paradigms including "
            "object-oriented and functional programming.",
            "What is Python programming language?",
        )
        assert score >= 0.8

    def test_single_keyword_question(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Federalism divides power between national and state governments.",
            "Federalism?",
        )
        assert score == 1.0


# ---------------------------------------------------------------------------
# score() — fully irrelevant
# ---------------------------------------------------------------------------

class TestIrrelevantAnswer:
    """Tests where the answer is NOT relevant to the question."""

    def test_completely_unrelated(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Quantum entanglement connects distant particles instantly.",
            "What is federalism?",
        )
        assert score == 0.0

    def test_no_keyword_overlap(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Cats sleep for sixteen hours per day.",
            "How does photosynthesis convert sunlight into energy?",
        )
        assert score == 0.0

    def test_empty_answer(self):
        scorer = RelevanceScorer()
        score = scorer.score("", "What is X?")
        assert score == 0.0

    def test_whitespace_answer(self):
        scorer = RelevanceScorer()
        score = scorer.score("   ", "What is X?")
        assert score == 0.0

    def test_empty_question(self):
        scorer = RelevanceScorer()
        score = scorer.score("Some answer.", "")
        assert score == 0.0

    def test_whitespace_question(self):
        scorer = RelevanceScorer()
        score = scorer.score("Some answer.", "   ")
        assert score == 0.0

    def test_both_empty(self):
        scorer = RelevanceScorer()
        score = scorer.score("", "")
        assert score == 0.0


# ---------------------------------------------------------------------------
# score() — partial relevance
# ---------------------------------------------------------------------------

class TestPartialRelevance:
    """Tests where the answer partially addresses the question."""

    def test_some_keywords_present(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Federalism is a political concept.",
            "What is federalism and how does it divide governmental power?",
        )
        assert 0.0 < score < 1.0

    def test_score_between_0_and_1(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "Machine learning algorithms process data efficiently.",
            "How do neural networks improve deep learning accuracy?",
        )
        assert 0.0 <= score <= 1.0

    def test_half_keywords_matched(self):
        scorer = RelevanceScorer()
        # Question tokens: "blockchain", "decentralize", "financial", "transactions"
        # Answer has only "blockchain" and "transactions"
        score = scorer.score(
            "Blockchain records transactions in a ledger.",
            "How does blockchain decentralize financial transactions?",
        )
        assert 0.2 < score < 0.9


# ---------------------------------------------------------------------------
# score_detailed() — result structure
# ---------------------------------------------------------------------------

class TestScoreDetailed:
    """Tests for the detailed scoring result."""

    def test_returns_relevance_result(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Federalism divides power.", "What is federalism?"
        )
        assert isinstance(result, RelevanceResult)

    def test_question_tokens_populated(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Federalism divides power.", "What is federalism?"
        )
        assert "federalism" in result.question_tokens
        assert len(result.question_tokens) >= 1

    def test_matched_tokens_populated(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Federalism divides power.", "What is federalism?"
        )
        assert "federalism" in result.matched_tokens

    def test_unmatched_tokens_populated(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Cats sleep all day.",
            "What is federalism?",
        )
        assert "federalism" in result.unmatched_tokens

    def test_matched_plus_unmatched_equals_question_tokens(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Federalism divides power between governments.",
            "What is federalism and how does it divide governmental power?",
        )
        assert len(result.matched_tokens) + len(result.unmatched_tokens) == len(
            result.question_tokens
        )

    def test_empty_answer_detailed(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed("", "What is X?")
        assert result.score == 0.0
        assert result.question_tokens == []
        assert result.matched_tokens == []

    def test_empty_question_detailed(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed("Some answer.", "")
        assert result.score == 0.0

    def test_overall_score_matches_score_method(self):
        scorer = RelevanceScorer()
        answer = "Federalism divides power."
        question = "What is federalism?"
        assert scorer.score(answer, question) == scorer.score_detailed(
            answer, question
        ).score


# ---------------------------------------------------------------------------
# Length penalty
# ---------------------------------------------------------------------------

class TestLengthPenalty:
    """Tests for the length penalty behaviour."""

    def test_penalty_reduces_short_answer_score(self):
        scorer_with = RelevanceScorer(length_penalty=True)
        scorer_without = RelevanceScorer(length_penalty=False)
        answer = "Federalism."
        question = "What is federalism and how does it divide governmental power between national and state levels?"
        score_with = scorer_with.score(answer, question)
        score_without = scorer_without.score(answer, question)
        # With penalty should be lower or equal
        assert score_with <= score_without

    def test_no_penalty_when_disabled(self):
        scorer = RelevanceScorer(length_penalty=False)
        score = scorer.score(
            "Federalism.",
            "What is federalism and how does it divide governmental power?",
        )
        # Without penalty, even a short answer matching 1 keyword should score > 0
        assert score > 0.0

    def test_no_penalty_when_answer_long_enough(self):
        scorer_with = RelevanceScorer(length_penalty=True)
        scorer_without = RelevanceScorer(length_penalty=False)
        answer = "Federalism divides governmental power between national and state levels through a constitutional framework."
        question = "What is federalism?"
        # Answer has more tokens than question — no penalty applied
        assert scorer_with.score(answer, question) == scorer_without.score(
            answer, question
        )


# ---------------------------------------------------------------------------
# Stopword-only question
# ---------------------------------------------------------------------------

class TestStopwordOnlyQuestion:
    """Tests for questions that contain only stop-words."""

    def test_stopword_question_scores_1(self):
        scorer = RelevanceScorer()
        # "What is it?" — all tokens are stop-words or filler
        score = scorer.score("Some answer text.", "What is it?")
        assert score == 1.0

    def test_stopword_question_detailed(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed("Some answer.", "What is it?")
        assert result.score == 1.0
        assert result.question_tokens == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case tests."""

    def test_case_insensitive(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "FEDERALISM DIVIDES POWER.",
            "what is federalism?",
        )
        assert score == 1.0

    def test_numbers_in_question(self):
        scorer = RelevanceScorer()
        score = scorer.score(
            "The Constitution has 27 amendments adopted since 1791.",
            "How many amendments does the Constitution have since 1791?",
        )
        assert score >= 0.5

    def test_question_filler_words_excluded(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Federalism divides power.",
            "Please explain federalism.",
        )
        # "please" and "explain" are in _QUESTION_FILLER — should not be in question_tokens
        assert "please" not in result.question_tokens
        assert "explain" not in result.question_tokens

    def test_duplicate_question_tokens_deduplicated(self):
        scorer = RelevanceScorer()
        result = scorer.score_detailed(
            "Federalism divides power.",
            "What is federalism? Tell me about federalism.",
        )
        # "federalism" appears twice but should be deduplicated
        assert result.question_tokens.count("federalism") == 1

    def test_single_word_answer_and_question(self):
        scorer = RelevanceScorer()
        score = scorer.score("Federalism.", "Federalism?")
        assert score == 1.0

    def test_very_long_answer(self):
        scorer = RelevanceScorer()
        long_answer = " ".join(
            [f"Word{i} appears in this very long generated answer." for i in range(100)]
        )
        score = scorer.score(long_answer, "What word appears in the answer?")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Data class immutability
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Tests for result dataclass properties."""

    def test_relevance_result_frozen(self):
        r = RelevanceResult(score=0.5)
        with pytest.raises(AttributeError):
            r.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify scorer is importable from the scorers package."""

    def test_scorer_exported(self):
        from src.eval.scorers import RelevanceScorer as cls
        assert cls is not None

    def test_result_exported(self):
        from src.eval.scorers import RelevanceResult
        assert RelevanceResult is not None
