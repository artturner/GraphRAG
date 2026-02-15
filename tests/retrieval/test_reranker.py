"""Tests for the reranker implementations."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.exceptions import RetrievalError
from src.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    IdentityReranker,
)
from src.store.base import SearchResult
from src.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_id: str = "chunk-001",
    score: float = 0.9,
    content: str = "Some content",
    source: str = "doc.txt",
) -> SearchResult:
    chunk = Chunk(
        id=chunk_id,
        document_id="doc-001",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={"source": source},
    )
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=chunk,
        metadata={"source": source},
    )


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        _make_result("c-1", 0.95, "Federalism is a system of government."),
        _make_result("c-2", 0.85, "The Constitution has three branches."),
        _make_result("c-3", 0.70, "Separation of powers prevents tyranny."),
    ]


# ---------------------------------------------------------------------------
# BaseReranker
# ---------------------------------------------------------------------------

class TestBaseReranker:
    """Tests for the BaseReranker abstract class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_subclass_must_implement_rerank(self):
        class BadReranker(BaseReranker):
            pass

        with pytest.raises(TypeError):
            BadReranker()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# IdentityReranker
# ---------------------------------------------------------------------------

class TestIdentityReranker:
    """Tests for the no-op IdentityReranker."""

    def test_is_subclass_of_base(self):
        assert issubclass(IdentityReranker, BaseReranker)

    def test_returns_same_order(self, sample_results: list[SearchResult]):
        reranker = IdentityReranker()
        reranked = reranker.rerank("What is federalism?", sample_results)

        assert len(reranked) == len(sample_results)
        for original, returned in zip(sample_results, reranked):
            assert original.chunk_id == returned.chunk_id
            assert original.score == returned.score

    def test_returns_copy_not_same_list(self, sample_results: list[SearchResult]):
        reranker = IdentityReranker()
        reranked = reranker.rerank("query", sample_results)
        assert reranked is not sample_results

    def test_empty_results(self):
        reranker = IdentityReranker()
        assert reranker.rerank("query", []) == []

    def test_single_result(self):
        reranker = IdentityReranker()
        results = [_make_result("c-1", 0.8)]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 1
        assert reranked[0].chunk_id == "c-1"


# ---------------------------------------------------------------------------
# CrossEncoderReranker — unit tests (mocked model)
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker with a mocked cross-encoder model."""

    def test_is_subclass_of_base(self):
        assert issubclass(CrossEncoderReranker, BaseReranker)

    def test_default_model_name(self):
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model_name(self):
        reranker = CrossEncoderReranker(model_name="my-model")
        assert reranker.model_name == "my-model"

    def test_repr(self):
        reranker = CrossEncoderReranker(model_name="my-model")
        assert "my-model" in repr(reranker)

    def test_empty_results(self):
        reranker = CrossEncoderReranker()
        # Should not even load the model
        assert reranker.rerank("query", []) == []

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_changes_order(
        self, mock_ce_cls: MagicMock, sample_results: list[SearchResult]
    ):
        """Cross-encoder scores should determine the new ordering."""
        mock_model = MagicMock()
        # Return scores that reverse the original order
        mock_model.predict.return_value = np.array([0.1, 0.5, 0.9])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("What is federalism?", sample_results)

        assert len(reranked) == 3
        # c-3 had the highest cross-encoder score (0.9)
        assert reranked[0].chunk_id == "c-3"
        # c-2 was second (0.5)
        assert reranked[1].chunk_id == "c-2"
        # c-1 was lowest (0.1)
        assert reranked[2].chunk_id == "c-1"

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_scores_normalised_to_zero_one(
        self, mock_ce_cls: MagicMock, sample_results: list[SearchResult]
    ):
        """Normalised scores should be in [0, 1]."""
        mock_model = MagicMock()
        # Raw logits can be arbitrary floats
        mock_model.predict.return_value = np.array([-3.0, 0.0, 5.0])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("query", sample_results)

        for r in reranked:
            assert 0.0 <= r.score <= 1.0

        # Highest raw score (5.0) should map to 1.0
        assert reranked[0].score == 1.0
        # Lowest raw score (-3.0) should map to 0.0
        assert reranked[-1].score == 0.0

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_identical_scores_normalise_to_one(
        self, mock_ce_cls: MagicMock, sample_results: list[SearchResult]
    ):
        """When all raw scores are equal, all normalised scores should be 1.0."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.0, 2.0, 2.0])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("query", sample_results)

        assert all(r.score == 1.0 for r in reranked)

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_preserves_metadata(
        self, mock_ce_cls: MagicMock, sample_results: list[SearchResult]
    ):
        """Chunk content and metadata must survive reranking."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5, 0.5])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("query", sample_results)

        original_ids = {r.chunk_id for r in sample_results}
        reranked_ids = {r.chunk_id for r in reranked}
        assert original_ids == reranked_ids

        for r in reranked:
            assert r.chunk.content  # non-empty
            assert r.metadata.get("source") == "doc.txt"

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_single_result(self, mock_ce_cls: MagicMock):
        """A single result should be returned with score 1.0."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.7])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = [_make_result("c-1", 0.6)]
        reranked = reranker.rerank("query", results)

        assert len(reranked) == 1
        assert reranked[0].chunk_id == "c-1"
        assert reranked[0].score == 1.0

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_passes_correct_pairs_to_model(
        self, mock_ce_cls: MagicMock, sample_results: list[SearchResult]
    ):
        """The model should receive (query, passage) pairs."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.2, 0.3])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranker.rerank("What is federalism?", sample_results)

        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert len(pairs) == 3
        for pair in pairs:
            assert pair[0] == "What is federalism?"
            assert isinstance(pair[1], str)


# ---------------------------------------------------------------------------
# CrossEncoderReranker — error handling
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerErrors:
    """Tests for error handling in CrossEncoderReranker."""

    @patch(
        "src.retrieval.reranker.CrossEncoder",
        side_effect=Exception("Model not found"),
    )
    def test_model_load_failure(self, mock_ce_cls: MagicMock):
        reranker = CrossEncoderReranker()
        results = [_make_result()]

        with pytest.raises(RetrievalError, match="Failed to load cross-encoder"):
            reranker.rerank("query", results)

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_predict_failure(
        self, mock_ce_cls: MagicMock, sample_results: list[SearchResult]
    ):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()

        with pytest.raises(RetrievalError, match="reranking failed"):
            reranker.rerank("query", sample_results)

    def test_missing_sentence_transformers(self):
        reranker = CrossEncoderReranker()
        results = [_make_result()]

        with patch("src.retrieval.reranker.CrossEncoder", None):
            reranker._model = None  # force re-load
            with pytest.raises(RetrievalError, match="sentence-transformers"):
                reranker.rerank("query", results)
