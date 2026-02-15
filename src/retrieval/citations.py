"""Citation extraction and formatting from search results.

This module provides the CitationBuilder class for converting search results
into structured citations and formatting them in various styles.
"""

import logging
from pathlib import PurePosixPath, PureWindowsPath

from src.store.base import SearchResult
from src.types import Citation

logger = logging.getLogger(__name__)


def _source_basename(source: str) -> str:
    """Extract the filename from a source path or URL."""
    for cls in (PureWindowsPath, PurePosixPath):
        try:
            return cls(source).name
        except Exception:
            continue
    return source


class CitationBuilder:
    """Builds and formats citations from vector store search results.

    The builder converts SearchResult objects into Citation models and
    supports formatting them in multiple styles (default, MLA, APA).

    Example:
        ```python
        builder = CitationBuilder()
        citations = builder.build_citations(search_results)
        for citation in citations:
            print(builder.format_citation(citation, style="apa"))
        ```
    """

    # Supported formatting styles
    SUPPORTED_STYLES = ("default", "mla", "apa")

    def build_citations(self, results: list[SearchResult]) -> list[Citation]:
        """Convert search results into a list of Citation objects.

        Each search result is mapped to a citation using the chunk's
        metadata and content. Duplicate chunk IDs are skipped.

        Args:
            results: Search results from a vector store query.

        Returns:
            A deduplicated list of Citation objects ordered by score
            (highest first).
        """
        seen_ids: set[str] = set()
        citations: list[Citation] = []

        for result in results:
            if result.chunk_id in seen_ids:
                continue
            seen_ids.add(result.chunk_id)

            source = (
                result.metadata.get("source")
                or result.chunk.metadata.get("source")
                or "unknown"
            )

            citation = Citation(
                source=source,
                chunk_id=result.chunk_id,
                text=result.chunk.content,
                score=result.score,
            )
            citations.append(citation)

        logger.debug("Built %d citations from %d results", len(citations), len(results))
        return citations

    def format_citation(self, citation: Citation, style: str = "default") -> str:
        """Format a citation as a human-readable string.

        Args:
            citation: The citation to format.
            style: Formatting style — one of "default", "mla", or "apa".

        Returns:
            A formatted citation string.

        Raises:
            ValueError: If the style is not supported.
        """
        style = style.lower()
        if style not in self.SUPPORTED_STYLES:
            raise ValueError(
                f"Unsupported citation style '{style}'. "
                f"Supported styles: {', '.join(self.SUPPORTED_STYLES)}"
            )

        if style == "mla":
            return self._format_mla(citation)
        elif style == "apa":
            return self._format_apa(citation)
        return self._format_default(citation)

    # ------------------------------------------------------------------
    # Private formatting helpers
    # ------------------------------------------------------------------

    def _format_default(self, citation: Citation) -> str:
        """Format in the default inline style.

        Example output:
            [chunk-001] (score: 0.95) document.txt — "The relevant passage..."
        """
        filename = _source_basename(citation.source)
        snippet = self._truncate(citation.text, 100)
        return (
            f"[{citation.chunk_id}] (score: {citation.score:.2f}) "
            f"{filename} \u2014 \"{snippet}\""
        )

    def _format_mla(self, citation: Citation) -> str:
        """Format in MLA-like style.

        Example output:
            "The relevant passage..." (document.txt, chunk-001).
        """
        filename = _source_basename(citation.source)
        snippet = self._truncate(citation.text, 120)
        return f"\"{snippet}\" ({filename}, {citation.chunk_id})."

    def _format_apa(self, citation: Citation) -> str:
        """Format in APA-like style.

        Example output:
            document.txt (chunk-001, relevance: 0.95). "The relevant passage..."
        """
        filename = _source_basename(citation.source)
        snippet = self._truncate(citation.text, 120)
        return (
            f"{filename} ({citation.chunk_id}, relevance: {citation.score:.2f}). "
            f"\"{snippet}\""
        )

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Truncate text to *max_length* characters, appending '...' if needed."""
        text = " ".join(text.split())  # normalise whitespace
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."
