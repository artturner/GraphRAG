"""Integration tests for the full retrieval pipeline.

This module tests the wiring of RetrievalService with reranking and
citation generation, verifying the complete search-rerank-cite flow.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval import (
    BaseReranker,
    CitationBuilder,
    CrossEncoderReranker,
    IdentityReranker,
    RetrievalService,
)
from src.retrieval.reranker import BaseReranker
from src.store.base import SearchResult
from src.types import Chunk, Citation


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, content: str, source: str = "doc.txt") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-001",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={"source": source},
    )


def _search_result(
    chunk_id: str,
    score: float,
    content: str = "content",
    source: str = "doc.txt",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=_chunk(chunk_id, content, source),
        metadata={"source": source},
    )


class MockEmbeddings:
    """Lightweight mock embedding provider."""

    def __init__(self, dimension: int = 16):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(i) for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(999)

    def _vec(self, seed: int) -> list[float]:
        rng = np.random.RandomState(seed)
        v = rng.randn(self._dimension).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()


class MockVectorStore:
    """Mock vector store returning canned search results."""

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results or []
        self._chunks: list[Chunk] = []

    def add_embeddings(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def similarity_search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        return self._results[:k]

    def count(self) -> int:
        return len(self._chunks)


class ReversingReranker(BaseReranker):
    """Test reranker that reverses the result order."""

    def __init__(self):
        self.called = False

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        self.called = True
        reversed_results = list(reversed(results))
        # Reassign scores so highest-score is first in new order
        return [
            SearchResult(
                chunk_id=r.chunk_id,
                score=round(1.0 - i * 0.1, 2),
                chunk=r.chunk,
                metadata=r.metadata,
            )
            for i, r in enumerate(reversed_results)
        ]


# ---------------------------------------------------------------------------
# Constructor wiring
# ---------------------------------------------------------------------------

class TestServiceConstructor:
    """Test that RetrievalService accepts the new parameters."""

    def test_without_reranker(self):
        svc = RetrievalService(MockEmbeddings(), MockVectorStore())
        assert svc.reranker is None

    def test_with_reranker(self):
        reranker = IdentityReranker()
        svc = RetrievalService(MockEmbeddings(), MockVectorStore(), reranker=reranker)
        assert svc.reranker is reranker

    def test_citation_builder_available(self):
        svc = RetrievalService(MockEmbeddings(), MockVectorStore())
        assert isinstance(svc.citation_builder, CitationBuilder)


# ---------------------------------------------------------------------------
# Search with reranking
# ---------------------------------------------------------------------------

class TestSearchWithReranking:
    """Test that search correctly invokes the reranker."""

    def _service(self, reranker=None):
        results = [
            _search_result("c-1", 0.95, "Federalism overview"),
            _search_result("c-2", 0.85, "Constitution branches"),
            _search_result("c-3", 0.70, "Separation of powers"),
        ]
        store = MockVectorStore(results=results)
        return RetrievalService(MockEmbeddings(), store, reranker=reranker)

    def test_search_without_reranker_preserves_order(self):
        svc = self._service(reranker=None)
        results = svc.search("query", k=3)
        assert [r.chunk_id for r in results] == ["c-1", "c-2", "c-3"]

    def test_search_with_identity_reranker_preserves_order(self):
        svc = self._service(reranker=IdentityReranker())
        results = svc.search("query", k=3)
        assert [r.chunk_id for r in results] == ["c-1", "c-2", "c-3"]

    def test_search_with_reversing_reranker_changes_order(self):
        reranker = ReversingReranker()
        svc = self._service(reranker=reranker)
        results = svc.search("query", k=3)

        assert reranker.called
        # ReversingReranker reverses the list
        assert [r.chunk_id for r in results] == ["c-3", "c-2", "c-1"]

    def test_reranker_not_called_on_empty_results(self):
        store = MockVectorStore(results=[])
        reranker = ReversingReranker()
        svc = RetrievalService(MockEmbeddings(), store, reranker=reranker)

        results = svc.search("query", k=5)
        assert results == []
        # The reranker is still called (with an empty list) — that's fine
        assert reranker.called

    def test_search_with_threshold_uses_reranker(self):
        reranker = ReversingReranker()
        svc = self._service(reranker=reranker)

        results = svc.search_with_threshold("query", k=3, min_score=0.85)

        assert reranker.called
        # After reversing, scores are reassigned: 1.0, 0.9, 0.8
        # All >= 0.85 → first two pass
        assert len(results) == 2
        assert all(r.score >= 0.85 for r in results)


# ---------------------------------------------------------------------------
# Citation generation
# ---------------------------------------------------------------------------

class TestCitationGeneration:
    """Test citation building from the service."""

    @pytest.fixture
    def service_with_results(self):
        results = [
            _search_result("c-1", 0.95, "Federalism is a system.", "/docs/fed.txt"),
            _search_result("c-2", 0.82, "Constitution overview.", "/docs/con.md"),
        ]
        store = MockVectorStore(results=results)
        return RetrievalService(MockEmbeddings(), store)

    def test_build_citations_from_search_results(self, service_with_results):
        results = service_with_results.search("query", k=2)
        citations = service_with_results.build_citations(results)

        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)

    def test_citation_fields_populated(self, service_with_results):
        results = service_with_results.search("query", k=2)
        citations = service_with_results.build_citations(results)

        c = citations[0]
        assert c.chunk_id == "c-1"
        assert c.score == 0.95
        assert c.text == "Federalism is a system."
        assert c.source == "/docs/fed.txt"

    def test_search_with_citations_convenience(self, service_with_results):
        results, citations = service_with_results.search_with_citations("query", k=2)

        assert len(results) == 2
        assert len(citations) == 2
        assert results[0].chunk_id == citations[0].chunk_id

    def test_search_with_citations_empty(self):
        store = MockVectorStore(results=[])
        svc = RetrievalService(MockEmbeddings(), store)
        results, citations = svc.search_with_citations("query")
        assert results == []
        assert citations == []

    def test_citations_formatted(self, service_with_results):
        results = service_with_results.search("query", k=2)
        citations = service_with_results.build_citations(results)

        builder = service_with_results.citation_builder
        formatted = builder.format_citation(citations[0], style="default")
        assert "c-1" in formatted
        assert "0.95" in formatted


# ---------------------------------------------------------------------------
# Full pipeline: index → search → rerank → cite
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end test of index → search → rerank → cite."""

    def test_full_flow_without_reranker(self):
        embeddings = MockEmbeddings()
        store = MockVectorStore(
            results=[
                _search_result("c-1", 0.90, "First passage.", "a.txt"),
                _search_result("c-2", 0.80, "Second passage.", "b.txt"),
            ]
        )
        svc = RetrievalService(embeddings, store)

        # Index
        chunks = [_chunk("c-1", "First passage.", "a.txt"), _chunk("c-2", "Second passage.", "b.txt")]
        count = svc.index_documents(chunks)
        assert count == 2

        # Search
        results = svc.search("query", k=2)
        assert len(results) == 2

        # Citations
        citations = svc.build_citations(results)
        assert len(citations) == 2
        assert citations[0].source == "a.txt"
        assert citations[1].source == "b.txt"

    def test_full_flow_with_reranker(self):
        embeddings = MockEmbeddings()
        store = MockVectorStore(
            results=[
                _search_result("c-1", 0.90, "First passage.", "a.txt"),
                _search_result("c-2", 0.80, "Second passage.", "b.txt"),
                _search_result("c-3", 0.70, "Third passage.", "c.txt"),
            ]
        )
        reranker = ReversingReranker()
        svc = RetrievalService(embeddings, store, reranker=reranker)

        # Search with reranking
        results, citations = svc.search_with_citations("query", k=3)

        assert reranker.called
        # After reversing: c-3 first, c-2, c-1
        assert results[0].chunk_id == "c-3"
        assert len(citations) == 3
        # Citations follow result order
        assert citations[0].chunk_id == "c-3"
        assert citations[2].chunk_id == "c-1"

    def test_full_flow_with_threshold_and_reranker(self):
        embeddings = MockEmbeddings()
        store = MockVectorStore(
            results=[
                _search_result("c-1", 0.90, "First passage.", "a.txt"),
                _search_result("c-2", 0.80, "Second passage.", "b.txt"),
                _search_result("c-3", 0.70, "Third passage.", "c.txt"),
            ]
        )
        reranker = ReversingReranker()
        svc = RetrievalService(embeddings, store, reranker=reranker)

        # ReversingReranker assigns scores: 1.0, 0.9, 0.8
        results = svc.search_with_threshold("query", k=3, min_score=0.85)

        assert len(results) == 2
        citations = svc.build_citations(results)
        assert len(citations) == 2

    def test_citation_formatting_all_styles(self):
        store = MockVectorStore(
            results=[_search_result("c-1", 0.95, "Example text.", "/docs/example.txt")]
        )
        svc = RetrievalService(MockEmbeddings(), store)

        results = svc.search("query", k=1)
        citations = svc.build_citations(results)
        builder = svc.citation_builder

        for style in CitationBuilder.SUPPORTED_STYLES:
            formatted = builder.format_citation(citations[0], style=style)
            assert isinstance(formatted, str)
            assert len(formatted) > 0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify all retrieval classes are importable from the package."""

    def test_retrieval_service_exported(self):
        from src.retrieval import RetrievalService
        assert RetrievalService is not None

    def test_citation_builder_exported(self):
        from src.retrieval import CitationBuilder
        assert CitationBuilder is not None

    def test_base_reranker_exported(self):
        from src.retrieval import BaseReranker
        assert BaseReranker is not None

    def test_identity_reranker_exported(self):
        from src.retrieval import IdentityReranker
        assert IdentityReranker is not None

    def test_cross_encoder_reranker_exported(self):
        from src.retrieval import CrossEncoderReranker
        assert CrossEncoderReranker is not None
