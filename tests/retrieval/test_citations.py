"""Tests for the CitationBuilder class."""

import pytest

from src.retrieval.citations import CitationBuilder
from src.store.base import SearchResult
from src.types import Chunk, Citation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "chunk-001",
    document_id: str = "doc-001",
    content: str = "Federalism is a system of government.",
    source: str | None = None,
) -> Chunk:
    metadata = {}
    if source is not None:
        metadata["source"] = source
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata=metadata,
    )


def _make_result(
    chunk_id: str = "chunk-001",
    score: float = 0.95,
    content: str = "Federalism is a system of government.",
    source: str | None = None,
    result_meta_source: str | None = None,
) -> SearchResult:
    chunk = _make_chunk(chunk_id=chunk_id, content=content, source=source)
    metadata = {}
    if result_meta_source is not None:
        metadata["source"] = result_meta_source
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        chunk=chunk,
        metadata=metadata,
    )


@pytest.fixture
def builder() -> CitationBuilder:
    return CitationBuilder()


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        _make_result(
            chunk_id="chunk-001",
            score=0.95,
            content="Federalism is a system of government.",
            result_meta_source="/docs/federalism.txt",
        ),
        _make_result(
            chunk_id="chunk-002",
            score=0.82,
            content="The Constitution establishes three branches.",
            result_meta_source="/docs/constitution.md",
        ),
        _make_result(
            chunk_id="chunk-003",
            score=0.71,
            content="Separation of powers prevents tyranny.",
            result_meta_source="/docs/separation.pdf",
        ),
    ]


# ---------------------------------------------------------------------------
# build_citations
# ---------------------------------------------------------------------------

class TestBuildCitations:
    """Tests for CitationBuilder.build_citations."""

    def test_builds_citations_from_results(
        self, builder: CitationBuilder, sample_results: list[SearchResult]
    ):
        citations = builder.build_citations(sample_results)
        assert len(citations) == 3
        assert all(isinstance(c, Citation) for c in citations)

    def test_preserves_order_by_score(
        self, builder: CitationBuilder, sample_results: list[SearchResult]
    ):
        citations = builder.build_citations(sample_results)
        assert citations[0].score == 0.95
        assert citations[1].score == 0.82
        assert citations[2].score == 0.71

    def test_maps_chunk_id(self, builder: CitationBuilder, sample_results: list[SearchResult]):
        citations = builder.build_citations(sample_results)
        assert [c.chunk_id for c in citations] == [
            "chunk-001",
            "chunk-002",
            "chunk-003",
        ]

    def test_maps_text_from_chunk_content(self, builder: CitationBuilder):
        results = [_make_result(content="Hello world", result_meta_source="a.txt")]
        citations = builder.build_citations(results)
        assert citations[0].text == "Hello world"

    def test_source_from_result_metadata(self, builder: CitationBuilder):
        results = [_make_result(result_meta_source="/path/doc.txt")]
        citations = builder.build_citations(results)
        assert citations[0].source == "/path/doc.txt"

    def test_source_falls_back_to_chunk_metadata(self, builder: CitationBuilder):
        results = [_make_result(source="/chunk/source.txt")]
        citations = builder.build_citations(results)
        assert citations[0].source == "/chunk/source.txt"

    def test_source_defaults_to_unknown(self, builder: CitationBuilder):
        results = [_make_result()]
        citations = builder.build_citations(results)
        assert citations[0].source == "unknown"

    def test_deduplicates_by_chunk_id(self, builder: CitationBuilder):
        r1 = _make_result(chunk_id="dup", score=0.9, result_meta_source="a.txt")
        r2 = _make_result(chunk_id="dup", score=0.8, result_meta_source="a.txt")
        citations = builder.build_citations([r1, r2])
        assert len(citations) == 1
        assert citations[0].score == 0.9

    def test_empty_results(self, builder: CitationBuilder):
        citations = builder.build_citations([])
        assert citations == []

    def test_single_result(self, builder: CitationBuilder):
        results = [_make_result(score=0.50, result_meta_source="file.md")]
        citations = builder.build_citations(results)
        assert len(citations) == 1
        assert citations[0].score == 0.50


# ---------------------------------------------------------------------------
# format_citation — default style
# ---------------------------------------------------------------------------

class TestFormatDefault:
    """Tests for the default citation style."""

    def test_contains_chunk_id(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Hello", score=0.9)
        out = builder.format_citation(c)
        assert "[c-1]" in out

    def test_contains_score(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Hello", score=0.85)
        out = builder.format_citation(c)
        assert "0.85" in out

    def test_contains_filename(self, builder: CitationBuilder):
        c = Citation(source="/a/b/doc.txt", chunk_id="c-1", text="Hello", score=0.9)
        out = builder.format_citation(c)
        assert "doc.txt" in out

    def test_contains_text_snippet(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Some content here", score=0.9)
        out = builder.format_citation(c)
        assert "Some content here" in out

    def test_truncates_long_text(self, builder: CitationBuilder):
        long_text = "A" * 200
        c = Citation(source="doc.txt", chunk_id="c-1", text=long_text, score=0.9)
        out = builder.format_citation(c)
        assert "..." in out
        # The full 200-char text should NOT appear
        assert long_text not in out


# ---------------------------------------------------------------------------
# format_citation — MLA style
# ---------------------------------------------------------------------------

class TestFormatMLA:
    """Tests for the MLA citation style."""

    def test_mla_starts_with_quoted_text(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Example text", score=0.9)
        out = builder.format_citation(c, style="mla")
        assert out.startswith('"Example text"')

    def test_mla_contains_filename(self, builder: CitationBuilder):
        c = Citation(source="/path/to/doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        out = builder.format_citation(c, style="mla")
        assert "doc.txt" in out

    def test_mla_contains_chunk_id(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        out = builder.format_citation(c, style="mla")
        assert "c-1" in out

    def test_mla_ends_with_period(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        out = builder.format_citation(c, style="mla")
        assert out.endswith(".")


# ---------------------------------------------------------------------------
# format_citation — APA style
# ---------------------------------------------------------------------------

class TestFormatAPA:
    """Tests for the APA citation style."""

    def test_apa_starts_with_filename(self, builder: CitationBuilder):
        c = Citation(source="/a/b/report.pdf", chunk_id="c-1", text="Txt", score=0.9)
        out = builder.format_citation(c, style="apa")
        assert out.startswith("report.pdf")

    def test_apa_contains_relevance_score(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.77)
        out = builder.format_citation(c, style="apa")
        assert "relevance: 0.77" in out

    def test_apa_contains_chunk_id(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        out = builder.format_citation(c, style="apa")
        assert "c-1" in out

    def test_apa_contains_quoted_text(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Some passage", score=0.9)
        out = builder.format_citation(c, style="apa")
        assert '"Some passage"' in out


# ---------------------------------------------------------------------------
# format_citation — style handling
# ---------------------------------------------------------------------------

class TestStyleHandling:
    """Tests for style validation and case handling."""

    def test_unsupported_style_raises(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        with pytest.raises(ValueError, match="Unsupported citation style"):
            builder.format_citation(c, style="chicago")

    def test_style_is_case_insensitive(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        out_lower = builder.format_citation(c, style="mla")
        out_upper = builder.format_citation(c, style="MLA")
        assert out_lower == out_upper

    def test_default_style_parameter(self, builder: CitationBuilder):
        c = Citation(source="doc.txt", chunk_id="c-1", text="Txt", score=0.9)
        out_default = builder.format_citation(c)
        out_explicit = builder.format_citation(c, style="default")
        assert out_default == out_explicit


# ---------------------------------------------------------------------------
# Citations with missing / minimal metadata
# ---------------------------------------------------------------------------

class TestMissingMetadata:
    """Tests for citation building when metadata is sparse."""

    def test_no_source_in_any_metadata(self, builder: CitationBuilder):
        chunk = Chunk(
            id="c-1",
            document_id="d-1",
            content="Orphan chunk",
            start_idx=0,
            end_idx=12,
            metadata={},
        )
        result = SearchResult(chunk_id="c-1", score=0.6, chunk=chunk, metadata={})
        citations = builder.build_citations([result])
        assert citations[0].source == "unknown"

    def test_format_citation_with_unknown_source(self, builder: CitationBuilder):
        c = Citation(source="unknown", chunk_id="c-1", text="Content", score=0.5)
        for style in CitationBuilder.SUPPORTED_STYLES:
            out = builder.format_citation(c, style=style)
            assert "unknown" in out

    def test_empty_chunk_content(self, builder: CitationBuilder):
        chunk = Chunk(
            id="c-1",
            document_id="d-1",
            content="",
            start_idx=0,
            end_idx=0,
            metadata={},
        )
        result = SearchResult(chunk_id="c-1", score=0.5, chunk=chunk, metadata={})
        citations = builder.build_citations([result])
        assert citations[0].text == ""

    def test_result_meta_takes_precedence_over_chunk_meta(self, builder: CitationBuilder):
        chunk = Chunk(
            id="c-1",
            document_id="d-1",
            content="Text",
            start_idx=0,
            end_idx=4,
            metadata={"source": "chunk_source.txt"},
        )
        result = SearchResult(
            chunk_id="c-1",
            score=0.9,
            chunk=chunk,
            metadata={"source": "result_source.txt"},
        )
        citations = builder.build_citations([result])
        assert citations[0].source == "result_source.txt"
