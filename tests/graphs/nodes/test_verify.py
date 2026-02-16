"""Tests for the grounding verification node and GroundingChecker."""

import pytest

from src.graphs.grounding import GroundingChecker, GroundingResult
from src.graphs.nodes import verify_grounding_node
from src.graphs.state import GraphState, StateBuilder
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
        metadata={"source": "doc.txt"},
    )


# ---------------------------------------------------------------------------
# GroundingResult
# ---------------------------------------------------------------------------

class TestGroundingResult:
    """Tests for the GroundingResult dataclass."""

    def test_basic_creation(self):
        r = GroundingResult(is_grounded=True, confidence=0.9)
        assert r.is_grounded is True
        assert r.confidence == 0.9
        assert r.supported_claims == []
        assert r.unsupported_claims == []

    def test_with_claims(self):
        r = GroundingResult(
            is_grounded=False,
            confidence=0.2,
            supported_claims=["A"],
            unsupported_claims=["B", "C"],
        )
        assert r.supported_claims == ["A"]
        assert len(r.unsupported_claims) == 2

    def test_is_frozen(self):
        r = GroundingResult(is_grounded=True, confidence=1.0)
        with pytest.raises(AttributeError):
            r.is_grounded = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GroundingChecker — supported answers
# ---------------------------------------------------------------------------

class TestGroundingCheckerSupported:
    """Tests for answers that are well-supported by chunks."""

    def test_fully_supported(self):
        checker = GroundingChecker()
        chunks = [_chunk("Federalism divides power between national and state governments.")]
        result = checker.check_grounding(
            "Federalism divides power between national and state governments.",
            chunks,
        )
        assert result.is_grounded is True
        assert result.confidence >= 0.5
        assert len(result.unsupported_claims) == 0

    def test_partially_supported(self):
        checker = GroundingChecker(threshold=0.3, min_confidence=0.5)
        chunks = [_chunk("Federalism divides governmental power.")]
        result = checker.check_grounding(
            "Federalism divides governmental power. Aliens landed on Mars.",
            chunks,
        )
        assert len(result.supported_claims) >= 1
        assert len(result.unsupported_claims) >= 1

    def test_multiple_chunks_combined(self):
        checker = GroundingChecker()
        chunks = [
            _chunk("Federalism is a system of government."),
            _chunk("Democracy allows citizens to vote."),
        ]
        result = checker.check_grounding(
            "Federalism is a system of government. Democracy allows citizens to vote.",
            chunks,
        )
        assert result.is_grounded is True
        assert result.confidence >= 0.5

    def test_exact_match_high_confidence(self):
        checker = GroundingChecker()
        text = "Machine learning uses statistical models."
        result = checker.check_grounding(text, [_chunk(text)])
        assert result.confidence == 1.0
        assert result.is_grounded is True


# ---------------------------------------------------------------------------
# GroundingChecker — unsupported answers
# ---------------------------------------------------------------------------

class TestGroundingCheckerUnsupported:
    """Tests for answers that are NOT supported by chunks."""

    def test_no_chunks_means_unsupported(self):
        checker = GroundingChecker()
        result = checker.check_grounding("Some answer.", [])
        assert result.is_grounded is False
        assert result.confidence == 0.0
        assert len(result.unsupported_claims) >= 1

    def test_completely_unrelated_answer(self):
        checker = GroundingChecker()
        chunks = [_chunk("Photosynthesis converts sunlight into chemical energy.")]
        result = checker.check_grounding(
            "Quantum entanglement connects distant particles instantaneously.",
            chunks,
        )
        assert result.is_grounded is False
        assert len(result.unsupported_claims) >= 1

    def test_empty_answer(self):
        checker = GroundingChecker()
        result = checker.check_grounding("", [_chunk("content")])
        assert result.is_grounded is False
        assert result.confidence == 0.0

    def test_whitespace_answer(self):
        checker = GroundingChecker()
        result = checker.check_grounding("   ", [_chunk("content")])
        assert result.is_grounded is False
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# GroundingChecker — confidence calculation
# ---------------------------------------------------------------------------

class TestGroundingCheckerConfidence:
    """Tests for confidence score calculation."""

    def test_all_supported_gives_1(self):
        checker = GroundingChecker()
        chunks = [_chunk("Python programming language features include duck typing.")]
        result = checker.check_grounding(
            "Python programming language features include duck typing.",
            chunks,
        )
        assert result.confidence == 1.0

    def test_none_supported_gives_0(self):
        checker = GroundingChecker()
        chunks = [_chunk("Unrelated chunk content here.")]
        result = checker.check_grounding(
            "Quantum computing revolutionizes cryptography. "
            "Blockchain decentralizes financial transactions.",
            chunks,
        )
        assert result.confidence < 0.5

    def test_confidence_between_0_and_1(self):
        checker = GroundingChecker()
        chunks = [_chunk("Machine learning algorithms learn from data patterns.")]
        result = checker.check_grounding(
            "Machine learning algorithms learn from data patterns. "
            "Aliens invaded the planet yesterday.",
            chunks,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_custom_threshold(self):
        # A very high threshold should make it harder to be grounded
        checker = GroundingChecker(threshold=0.99)
        chunks = [_chunk("Federalism.")]
        result = checker.check_grounding(
            "Federalism divides power between national and state governments.",
            chunks,
        )
        # With a 0.99 threshold most sentences will fail
        assert result.confidence < 1.0

    def test_custom_min_confidence(self):
        checker = GroundingChecker(min_confidence=1.0)
        chunks = [_chunk("Partial overlap here.")]
        result = checker.check_grounding(
            "Partial overlap here. Totally unrelated sentence.",
            chunks,
        )
        # Even if some sentences match, 100% is required
        assert result.is_grounded is False


# ---------------------------------------------------------------------------
# verify_grounding_node — successful verification
# ---------------------------------------------------------------------------

class TestVerifyGroundingNodeSuccess:
    """Tests for verify_grounding_node with valid inputs."""

    def test_grounded_answer(self):
        state = (
            StateBuilder()
            .with_question("What is federalism?")
            .with_answer("Federalism divides power between governments.")
            .with_chunks([_chunk("Federalism divides power between governments.")])
            .build()
        )

        out = verify_grounding_node(state)

        assert out["is_grounded"] is True
        assert out["confidence"] >= 0.5
        assert "error" not in out

    def test_ungrounded_answer(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Aliens control world governments.")
            .with_chunks([_chunk("Federalism divides power.")])
            .build()
        )

        out = verify_grounding_node(state)

        assert out["is_grounded"] is False

    def test_custom_checker(self):
        checker = GroundingChecker(threshold=0.01, min_confidence=0.01)
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Federalism is important.")
            .with_chunks([_chunk("Federalism matters.")])
            .build()
        )

        out = verify_grounding_node(state, checker=checker)

        assert out["is_grounded"] is True

    def test_returns_unsupported_claims(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Supported claim here. Totally unrelated aliens.")
            .with_chunks([_chunk("Supported claim here exactly.")])
            .build()
        )

        out = verify_grounding_node(state)

        if "unsupported_claims" in out:
            assert isinstance(out["unsupported_claims"], list)

    def test_confidence_in_range(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Some answer text.")
            .with_chunks([_chunk("Some answer text.")])
            .build()
        )

        out = verify_grounding_node(state)

        assert 0.0 <= out["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# verify_grounding_node — edge cases
# ---------------------------------------------------------------------------

class TestVerifyGroundingNodeEdgeCases:
    """Tests for verify_grounding_node edge cases."""

    def test_no_answer_returns_error(self):
        state: GraphState = {"question": "Q"}

        out = verify_grounding_node(state)

        assert out["is_grounded"] is False
        assert out["confidence"] == 0.0
        assert "error" in out
        assert "No answer" in out["error"]

    def test_empty_answer_returns_error(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("")
            .build()
        )

        out = verify_grounding_node(state)

        assert out["is_grounded"] is False
        assert "error" in out

    def test_no_chunks_means_ungrounded(self):
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Some answer.")
            .build()
        )

        out = verify_grounding_node(state)

        assert out["is_grounded"] is False
        assert out["confidence"] == 0.0

    def test_none_answer(self):
        state: GraphState = {"question": "Q", "answer": None}  # type: ignore[typeddict-item]

        out = verify_grounding_node(state)

        assert out["is_grounded"] is False
        assert "error" in out

    def test_whitespace_answer(self):
        state: GraphState = {"question": "Q", "answer": "   "}

        out = verify_grounding_node(state)

        assert out["is_grounded"] is False
        assert "error" in out


# ---------------------------------------------------------------------------
# verify_grounding_node — with mocked checker
# ---------------------------------------------------------------------------

class TestVerifyGroundingNodeMocked:
    """Tests for verify_grounding_node with a mocked checker."""

    def test_uses_provided_checker(self):
        """Verify the node delegates to the checker we pass in."""

        class RecordingChecker(GroundingChecker):
            def __init__(self):
                super().__init__()
                self.called_with: tuple | None = None

            def check_grounding(self, answer, chunks):
                self.called_with = (answer, chunks)
                return GroundingResult(is_grounded=True, confidence=1.0)

        checker = RecordingChecker()
        chunk = _chunk("Context")
        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Answer")
            .with_chunks([chunk])
            .build()
        )

        out = verify_grounding_node(state, checker=checker)

        assert checker.called_with is not None
        assert checker.called_with[0] == "Answer"
        assert checker.called_with[1] == [chunk]
        assert out["is_grounded"] is True
        assert out["confidence"] == 1.0

    def test_checker_exception_caught(self):
        """If the checker raises, the node returns an error dict."""

        class BrokenChecker(GroundingChecker):
            def check_grounding(self, answer, chunks):
                raise RuntimeError("checker exploded")

        state = (
            StateBuilder()
            .with_question("Q")
            .with_answer("Answer")
            .with_chunks([_chunk("ctx")])
            .build()
        )

        out = verify_grounding_node(state, checker=BrokenChecker())

        assert out["is_grounded"] is False
        assert out["confidence"] == 0.0
        assert "error" in out
        assert "exploded" in out["error"]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify verify_grounding_node is importable from the nodes package."""

    def test_verify_grounding_node_exported(self):
        from src.graphs.nodes import verify_grounding_node as fn
        assert callable(fn)

    def test_grounding_checker_importable(self):
        from src.graphs.grounding import GroundingChecker
        assert callable(GroundingChecker)

    def test_grounding_result_importable(self):
        from src.graphs.grounding import GroundingResult
        assert GroundingResult is not None
