"""End-to-end tests for the full ingestion-to-query pipeline.

These tests exercise the complete flow: load documents from disk,
clean and chunk them, embed and index into a FAISS store, then run
queries through the LangGraph workflow using a fake LLM.

Uses a lightweight fake embedding provider (bag-of-words hashing) to
avoid downloading the full sentence-transformers model.

Marked with ``@pytest.mark.e2e`` — run with ``pytest --run-e2e``.
"""

import hashlib
import math
import os
import tempfile

import pytest

from src.config import (
    CorpusConfig,
    EmbeddingsConfig,
    GraphConfig,
    VectorStoreConfig,
)
from src.connectors.local import LocalConnector
from src.embeddings.base import BaseEmbeddings
from src.graphs.qna_graph import create_qna_graph
from src.ingestion.chunking import FixedSizeChunker
from src.ingestion.cleaning import TextCleaner
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.service import RetrievalService
from src.store.faiss_store import FAISSVectorStore
from src.types import Chunk, EmbeddingVector

from tests.e2e.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Fake embeddings (deterministic, no model download)
# ---------------------------------------------------------------------------

_DIM = 64


class FakeEmbeddings(BaseEmbeddings):
    """Lightweight bag-of-words embeddings for testing.

    Produces a deterministic vector for any text by hashing each
    lowercased word to a dimension bucket and incrementing that slot.
    The result is L2-normalised so FAISS inner-product search works.
    """

    @property
    def dimension(self) -> int:
        return _DIM

    def _embed(self, text: str) -> EmbeddingVector:
        vec = [0.0] * _DIM
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
            vec[idx] += 1.0
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> EmbeddingVector:
        return self._embed(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline(corpus_dir: str, tmp_dir: str):
    """Build the full pipeline: ingest → embed → index → graph.

    Returns ``(graph, retrieval, chunks)`` where *graph* is the
    compiled LangGraph workflow.
    """
    # 1. Ingest
    connector = LocalConnector(source_path=corpus_dir)
    cleaner = TextCleaner()
    chunker = FixedSizeChunker(chunk_size=300, overlap=50)

    pipeline = IngestionPipeline(
        connector=connector,
        cleaner=cleaner,
        chunker=chunker,
    )
    result = pipeline.run()
    assert result.documents_count > 0, "No documents ingested"
    assert result.chunks_count > 0, "No chunks created"
    assert not result.errors, f"Ingestion errors: {result.errors}"

    # Re-run to get chunks (pipeline.run only returns counts)
    documents = connector.load()
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(pipeline.process_document(doc))

    # 2. Embed + index (using fake embeddings — no model download)
    embeddings = FakeEmbeddings()

    store = FAISSVectorStore(
        dimension=_DIM,
        persist_dir=os.path.join(tmp_dir, ".vectorstore"),
    )

    retrieval = RetrievalService(embeddings=embeddings, store=store)
    indexed = retrieval.index_documents(all_chunks)
    assert indexed > 0, "No chunks indexed"

    # 3. Build graph
    llm = FakeLLM()
    config = GraphConfig(max_retries=2, refusal_threshold=0.3)
    graph = create_qna_graph(retrieval=retrieval, llm=llm, config=config)

    return graph, retrieval, all_chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestIngestionFlow:
    """Verify the ingestion pipeline processes sample files correctly."""

    def test_ingest_creates_chunks(self, sample_corpus: str):
        """Documents should produce non-zero chunks."""
        connector = LocalConnector(source_path=sample_corpus)
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=TextCleaner(),
            chunker=FixedSizeChunker(chunk_size=300, overlap=50),
        )
        result = pipeline.run()

        assert result.documents_count == 3
        assert result.chunks_count > 0
        assert result.errors == []

    def test_ingest_progress_reports(self, sample_corpus: str):
        """run_with_progress should yield progress for each document."""
        connector = LocalConnector(source_path=sample_corpus)
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=TextCleaner(),
            chunker=FixedSizeChunker(chunk_size=300, overlap=50),
        )

        progress_list = list(pipeline.run_with_progress())
        assert len(progress_list) == 3
        final = progress_list[-1]
        assert final.documents_processed == 3
        assert final.chunks_created > 0

    def test_ingest_with_indexing(self, sample_corpus: str, tmp_dir: str):
        """Chunks should be indexable and searchable."""
        _, retrieval, chunks = _build_pipeline(sample_corpus, tmp_dir)

        assert retrieval.store.count() == len(chunks)
        results = retrieval.search("federalism", k=3)
        assert len(results) > 0


@pytest.mark.e2e
class TestFullQueryFlow:
    """Verify the full query pipeline from question to answer."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_corpus: str, tmp_dir: str):
        self.graph, self.retrieval, self.chunks = _build_pipeline(
            sample_corpus, tmp_dir,
        )

    def test_factual_question_returns_answer(self):
        """A factual question about corpus content should be answered."""
        result = self.graph.invoke({"question": "What is federalism?"})

        assert result.get("answer") is not None
        assert result.get("refusal_reason") is None or result.get("action") == "accept"
        assert result.get("query_type") == "factual"

    def test_answer_contains_relevant_content(self):
        """The answer should echo corpus content (via FakeLLM)."""
        result = self.graph.invoke({"question": "What is federalism?"})

        answer = result.get("answer", "")
        assert "federalism" in answer.lower() or "government" in answer.lower()

    def test_citations_are_populated(self):
        """The answer should include citations from retrieved chunks."""
        result = self.graph.invoke({"question": "What is federalism?"})

        citations = result.get("citations", [])
        assert len(citations) > 0

    def test_procedural_question(self):
        """Procedural questions should also produce answers."""
        result = self.graph.invoke(
            {"question": "How do I learn about the separation of powers?"}
        )

        assert result.get("query_type") == "procedural"
        # Should have an answer or refusal depending on FakeLLM
        assert result.get("answer") is not None or result.get("refusal_reason") is not None

    def test_unsupported_query_is_refused(self):
        """A greeting should be classified as unsupported and refused."""
        result = self.graph.invoke({"question": "Hello there!"})

        assert result.get("query_type") == "unsupported"
        assert result.get("refusal_reason") is not None
        assert result.get("answer") is None

    def test_confidence_is_set(self):
        """Accepted answers should carry a non-zero confidence."""
        result = self.graph.invoke({"question": "What is the Bill of Rights?"})

        if result.get("action") == "accept":
            assert result.get("confidence", 0) > 0

    def test_retry_count_is_tracked(self):
        """The retry count should be at least 1 (initial pass)."""
        result = self.graph.invoke({"question": "What is federalism?"})

        assert result.get("retry_count", 0) >= 1

    def test_chunks_in_state(self):
        """The graph state should contain the retrieved chunks."""
        result = self.graph.invoke({"question": "What is federalism?"})

        chunks = result.get("chunks", [])
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_search_results_in_state(self):
        """Search results should be populated in the state."""
        result = self.graph.invoke({"question": "What are the three branches of government?"})

        results = result.get("search_results", [])
        assert len(results) > 0

    def test_multiple_queries_sequential(self):
        """Multiple queries should work without interfering with each other."""
        q1 = self.graph.invoke({"question": "What is federalism?"})
        q2 = self.graph.invoke({"question": "What is the Bill of Rights?"})
        q3 = self.graph.invoke({"question": "Hello!"})

        # q1 and q2 should have answers, q3 should be refused
        assert q1.get("query_type") == "factual"
        assert q2.get("query_type") == "factual"
        assert q3.get("query_type") == "unsupported"
        assert q3.get("refusal_reason") is not None


@pytest.mark.e2e
class TestRetrievalQuality:
    """Verify that retrieval returns relevant chunks for queries."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_corpus: str, tmp_dir: str):
        _, self.retrieval, self.chunks = _build_pipeline(
            sample_corpus, tmp_dir,
        )

    def test_federalism_query_retrieves_relevant_chunks(self):
        """Searching for 'federalism' should return federalism-related content."""
        results = self.retrieval.search("federalism", k=3)

        assert len(results) > 0
        top_text = results[0].chunk.content.lower()
        assert "federal" in top_text or "government" in top_text

    def test_branches_query_retrieves_relevant_chunks(self):
        """Searching for branches of government should match."""
        results = self.retrieval.search("three branches of government", k=3)

        assert len(results) > 0
        all_text = " ".join(r.chunk.content.lower() for r in results)
        assert "legislative" in all_text or "congress" in all_text or "branch" in all_text

    def test_bill_of_rights_query(self):
        """Searching for Bill of Rights should find amendment-related content."""
        results = self.retrieval.search("Bill of Rights amendments", k=3)

        assert len(results) > 0
        all_text = " ".join(r.chunk.content.lower() for r in results)
        assert "amendment" in all_text or "rights" in all_text

    def test_search_returns_results_for_corpus_topics(self):
        """Queries about corpus topics should return non-empty results."""
        for query in ("federalism", "branches of government", "Bill of Rights"):
            results = self.retrieval.search(query, k=3)
            assert len(results) > 0, f"No results for: {query}"

    def test_search_with_citations(self):
        """search_with_citations should return both results and citations."""
        results, citations = self.retrieval.search_with_citations(
            "federalism", k=3,
        )

        assert len(results) > 0
        assert len(citations) > 0
        assert all(c.source for c in citations)
        assert all(c.chunk_id for c in citations)


@pytest.mark.e2e
class TestEdgeCases:
    """Edge cases for the full pipeline."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_corpus: str, tmp_dir: str):
        self.graph, self.retrieval, _ = _build_pipeline(
            sample_corpus, tmp_dir,
        )

    def test_very_long_question(self):
        """A very long question should still be processed."""
        long_q = "What is federalism? " * 50
        result = self.graph.invoke({"question": long_q})

        # Should not error out
        assert result.get("error") is None or result.get("refusal_reason") is not None

    def test_question_with_special_characters(self):
        """Questions with special chars should be handled."""
        result = self.graph.invoke(
            {"question": "What is the Bill of Rights? (1791)"}
        )

        assert result.get("query_type") == "factual"

    def test_empty_corpus_directory(self, tmp_dir: str):
        """An empty corpus should ingest zero documents."""
        empty_dir = os.path.join(tmp_dir, "empty_corpus")
        os.makedirs(empty_dir)

        connector = LocalConnector(source_path=empty_dir)
        pipeline = IngestionPipeline(
            connector=connector,
            cleaner=TextCleaner(),
            chunker=FixedSizeChunker(chunk_size=300, overlap=50),
        )
        result = pipeline.run()

        assert result.documents_count == 0
        assert result.chunks_count == 0
