"""Reranking support for improving retrieval quality.

This module provides reranker implementations that reorder search results
using more sophisticated scoring than the initial vector similarity search.

The CrossEncoderReranker uses a cross-encoder model that jointly encodes
the query and each candidate passage, producing more accurate relevance
scores at the cost of higher latency.
"""

import logging
from abc import ABC, abstractmethod

from src.exceptions import RetrievalError
from src.store.base import SearchResult

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment,misc]


class BaseReranker(ABC):
    """Abstract base class for reranker implementations.

    Rerankers take a query and a list of initial search results and
    return the same results reordered by a refined relevance score.

    Example:
        ```python
        class MyReranker(BaseReranker):
            def rerank(self, query, results):
                # custom reranking logic
                return sorted(results, key=lambda r: r.score, reverse=True)
        ```
    """

    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Rerank search results for the given query.

        Args:
            query: The original search query.
            results: Search results to rerank.

        Returns:
            The same results reordered by refined relevance, with
            updated scores normalised to [0, 1].
        """
        ...


class IdentityReranker(BaseReranker):
    """No-op reranker that returns results in their original order.

    Useful as a default when no reranking model is available, or for
    A/B testing the effect of reranking on retrieval quality.

    Example:
        ```python
        reranker = IdentityReranker()
        reranked = reranker.rerank("query", results)
        assert reranked == results
        ```
    """

    def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Return results unchanged."""
        return list(results)


class CrossEncoderReranker(BaseReranker):
    """Reranker that uses a cross-encoder model for pairwise scoring.

    Cross-encoders jointly encode (query, passage) pairs and produce a
    single relevance score, which is typically more accurate than the
    bi-encoder similarity used during initial retrieval.

    The model is loaded lazily on first use.

    Attributes:
        model_name: Name of the cross-encoder model.

    Example:
        ```python
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        reranked = reranker.rerank("What is federalism?", results)
        ```
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialise the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
            device: Device for inference (``None`` for auto-detect).
            batch_size: Batch size for scoring pairs.
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    @property
    def model_name(self) -> str:
        """Return the cross-encoder model name."""
        return self._model_name

    def _load_model(self) -> None:
        """Lazily load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            if CrossEncoder is None:
                raise ImportError("sentence-transformers is not installed")

            logger.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
            )
            logger.info("Cross-encoder model loaded successfully")
        except ImportError as e:
            raise RetrievalError(
                "sentence-transformers library not installed",
                details="Install with: pip install sentence-transformers",
            ) from e
        except Exception as e:
            raise RetrievalError(
                f"Failed to load cross-encoder model: {self._model_name}",
                details=str(e),
            ) from e

    def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Rerank results using the cross-encoder model.

        Each (query, passage) pair is scored jointly. The raw logits
        are normalised to [0, 1] via min-max scaling so that the
        returned ``SearchResult.score`` values remain in range.

        Args:
            query: The search query.
            results: Initial search results to rerank.

        Returns:
            Results sorted by cross-encoder score (highest first),
            with scores normalised to [0, 1].

        Raises:
            RetrievalError: If the model fails to load or score.
        """
        if not results:
            return []

        self._load_model()

        try:
            pairs = [[query, r.chunk.content] for r in results]
            raw_scores: list[float] = self._model.predict(  # type: ignore[union-attr]
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            ).tolist()

            # Min-max normalisation to [0, 1]
            min_s = min(raw_scores)
            max_s = max(raw_scores)
            if max_s - min_s > 0:
                norm_scores = [(s - min_s) / (max_s - min_s) for s in raw_scores]
            else:
                # All scores identical — assign uniform 1.0
                norm_scores = [1.0] * len(raw_scores)

            # Pair results with new scores and sort descending
            scored = sorted(
                zip(results, norm_scores),
                key=lambda pair: pair[1],
                reverse=True,
            )

            reranked = [
                SearchResult(
                    chunk_id=r.chunk_id,
                    score=round(score, 6),
                    chunk=r.chunk,
                    metadata=r.metadata,
                )
                for r, score in scored
            ]

            logger.debug(
                "Reranked %d results with cross-encoder", len(reranked)
            )
            return reranked

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(
                "Cross-encoder reranking failed",
                details=str(e),
            ) from e

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model_name={self._model_name!r})"
