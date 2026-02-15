"""Retrieval service that combines embeddings, vector store, and reranking.

This module provides the RetrievalService class which orchestrates
embedding generation, vector storage, optional reranking, and citation
extraction for document retrieval.
"""

import logging
from typing import Any

from src.embeddings.base import BaseEmbeddings
from src.exceptions import RetrievalError
from src.retrieval.citations import CitationBuilder
from src.retrieval.reranker import BaseReranker
from src.store.base import BaseVectorStore, SearchResult
from src.types import Chunk, Citation

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for indexing and retrieving documents using embeddings.

    This service combines an embedding provider with a vector store to
    provide a unified interface for document indexing and retrieval.

    The service handles:
    - Converting document chunks to embeddings
    - Storing embeddings in the vector store
    - Performing similarity searches
    - Optional reranking of search results
    - Citation extraction from search results
    - Filtering results by relevance threshold

    Attributes:
        embeddings: The embedding provider for generating vectors.
        store: The vector store for persisting and searching embeddings.
        reranker: Optional reranker for refining result ordering.

    Example:
        ```python
        from src.retrieval import RetrievalService, CrossEncoderReranker

        retrieval = RetrievalService(
            embeddings=embeddings,
            store=store,
            reranker=CrossEncoderReranker(),
        )
        results = retrieval.search("What is federalism?", k=5)
        citations = retrieval.build_citations(results)
        ```
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        store: BaseVectorStore,
        reranker: BaseReranker | None = None,
    ) -> None:
        """Initialize the retrieval service.

        Args:
            embeddings: The embedding provider for generating vectors.
            store: The vector store for persisting and searching embeddings.
            reranker: Optional reranker for refining search result ordering.
                If ``None``, results are returned in vector similarity order.

        Example:
            ```python
            embeddings = LocalEmbeddings()
            store = FAISSVectorStore(dimension=embeddings.dimension)
            reranker = CrossEncoderReranker()
            retrieval = RetrievalService(embeddings, store, reranker=reranker)
            ```
        """
        self._embeddings = embeddings
        self._store = store
        self._reranker = reranker
        self._citation_builder = CitationBuilder()

        logger.debug(
            "Initialized RetrievalService with embeddings dimension=%d, reranker=%s",
            embeddings.dimension,
            type(reranker).__name__ if reranker else "None",
        )

    @property
    def embeddings(self) -> BaseEmbeddings:
        """Return the embedding provider."""
        return self._embeddings

    @property
    def store(self) -> BaseVectorStore:
        """Return the vector store."""
        return self._store

    @property
    def reranker(self) -> BaseReranker | None:
        """Return the reranker, or ``None`` if not configured."""
        return self._reranker

    @property
    def citation_builder(self) -> CitationBuilder:
        """Return the citation builder instance."""
        return self._citation_builder

    def index_documents(self, chunks: list[Chunk]) -> int:
        """Index document chunks by generating and storing embeddings.

        This method:
        1. Extracts text content from each chunk
        2. Generates embeddings for all chunks in batch
        3. Stores the chunks and embeddings in the vector store

        Args:
            chunks: A list of Chunk objects to index.

        Returns:
            The number of chunks successfully indexed.

        Raises:
            RetrievalError: If embedding generation or storage fails.
            ValueError: If chunks is empty.

        Example:
            ```python
            chunks = [
                Chunk(
                    id="chunk-001",
                    document_id="doc-001",
                    content="Federalism is a system of government...",
                    start_idx=0,
                    end_idx=100,
                ),
            ]
            count = retrieval.index_documents(chunks)
            print(f"Indexed {count} chunks")
            ```
        """
        if not chunks:
            raise ValueError("Cannot index empty list of chunks")

        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]

            logger.info("Generating embeddings for %d chunks", len(chunks))

            # Generate embeddings in batch
            embedding_vectors = self._embeddings.embed_documents(texts)

            # Store chunks with their embeddings
            count = self._store.add_embeddings(chunks, embedding_vectors)

            logger.info("Successfully indexed %d chunks", count)

            return count

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error("Failed to index documents: %s", str(e))
            raise RetrievalError(
                "Failed to index documents",
                details=str(e),
            ) from e

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for documents similar to the query.

        This method:
        1. Generates an embedding for the query
        2. Performs a similarity search in the vector store
        3. Optionally reranks the results using the configured reranker
        4. Returns the top-k most similar results

        Args:
            query: The search query text.
            k: The maximum number of results to return. Defaults to 5.

        Returns:
            A list of SearchResult objects sorted by similarity score
            (highest first), with at most k results.

        Raises:
            RetrievalError: If the search operation fails.
            ValueError: If query is empty or k is not positive.

        Example:
            ```python
            results = retrieval.search("What is federalism?", k=5)
            for result in results:
                print(f"Score: {result.score:.3f}")
                print(f"Content: {result.chunk.content[:100]}...")
            ```
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be a positive integer")

        try:
            logger.debug("Searching for: %s (k=%d)", query[:50], k)

            # Generate embedding for the query
            query_embedding = self._embeddings.embed_query(query)

            # Perform similarity search
            results = self._store.similarity_search(query_embedding, k)

            # Apply reranking if configured
            if self._reranker is not None:
                logger.debug("Reranking %d results", len(results))
                results = self._reranker.rerank(query, results)

            logger.debug("Found %d results", len(results))

            return results

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error("Search failed: %s", str(e))
            raise RetrievalError(
                "Search operation failed",
                details=str(e),
            ) from e

    def search_with_citations(
        self, query: str, k: int = 5
    ) -> tuple[list[SearchResult], list[Citation]]:
        """Search and return both results and their citations.

        Convenience method that performs a search and immediately builds
        citations from the results.

        Args:
            query: The search query text.
            k: The maximum number of results to return.

        Returns:
            A tuple of (search_results, citations).

        Raises:
            RetrievalError: If the search operation fails.
            ValueError: If query is empty or k is not positive.

        Example:
            ```python
            results, citations = retrieval.search_with_citations(
                "What is federalism?", k=5
            )
            for citation in citations:
                print(retrieval.citation_builder.format_citation(citation))
            ```
        """
        results = self.search(query, k)
        citations = self._citation_builder.build_citations(results)
        return results, citations

    def build_citations(self, results: list[SearchResult]) -> list[Citation]:
        """Build citations from search results.

        Args:
            results: Search results to convert to citations.

        Returns:
            A list of Citation objects.

        Example:
            ```python
            results = retrieval.search("query")
            citations = retrieval.build_citations(results)
            ```
        """
        return self._citation_builder.build_citations(results)

    def search_with_threshold(
        self,
        query: str,
        k: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Search for documents with a minimum similarity threshold.

        This method performs a similarity search (with optional reranking)
        and filters the results to only include those with a score at or
        above the specified threshold.

        Args:
            query: The search query text.
            k: The maximum number of results to return before filtering.
            min_score: The minimum similarity score (0.0 to 1.0) required.

        Returns:
            A list of SearchResult objects with scores >= min_score,
            sorted by similarity score (highest first).

        Raises:
            RetrievalError: If the search operation fails.
            ValueError: If query is empty, k is not positive, or min_score
                       is not between 0.0 and 1.0.

        Example:
            ```python
            results = retrieval.search_with_threshold(
                "What is federalism?",
                k=10,
                min_score=0.8,
            )
            for result in results:
                print(f"Score: {result.score:.3f} (>= 0.8)")
            ```
        """
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")

        # Get all results from standard search (includes reranking)
        all_results = self.search(query, k)

        # Filter by threshold
        filtered_results = [
            result for result in all_results
            if result.score >= min_score
        ]

        logger.debug(
            "Filtered %d results to %d with min_score=%.3f",
            len(all_results),
            len(filtered_results),
            min_score,
        )

        return filtered_results
