"""Tests for the retry node."""

import pytest

from src.graphs.nodes import retry_node
from src.graphs.nodes.retry import DEFAULT_MAX_RETRIES, DEFAULT_MIN_CONFIDENCE
from src.graphs.state import GraphState, StateBuilder
from src.store.base import SearchResult
from src.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content: str = "text") -> Chunk:
    return Chunk(
        id="c-1",
        document_id="doc-1",
        content=content,
        start_idx=0,
        end_idx=len(content),
        metadata={},
    )


def _state(
    *,
    retry_count: int = 0,
    confidence: float = 0.0,
    answer: str | None = None,
    error: str | None = None,
) -> GraphState:
    builder = StateBuilder().with_question("Q").with_confidence(confidence).with_retry_count(retry_count)
    if answer is not None:
        builder = builder.with_answer(answer)
    if error is not None:
        builder = builder.with_error(error)
    return builder.build()


# ---------------------------------------------------------------------------
# Retry count increment
# ---------------------------------------------------------------------------

class TestRetryCountIncrement:
    """Tests that retry_count is always incremented."""

    def test_increments_from_zero(self):
        out = retry_node(_state(retry_count=0))
        assert out["retry_count"] == 1

    def test_increments_from_nonzero(self):
        out = retry_node(_state(retry_count=2))
        assert out["retry_count"] == 3

    def test_increments_on_accept(self):
        out = retry_node(_state(retry_count=0, confidence=0.9, answer="Good"))
        assert out["retry_count"] == 1

    def test_increments_on_refuse(self):
        out = retry_node(_state(retry_count=3, confidence=0.1))
        assert out["retry_count"] == 4

    def test_increments_when_missing_key(self):
        state: GraphState = {"question": "Q"}
        out = retry_node(state)
        assert out["retry_count"] == 1


# ---------------------------------------------------------------------------
# Retry decision — accept
# ---------------------------------------------------------------------------

class TestRetryAccept:
    """Tests for the 'accept' action."""

    def test_accept_when_confidence_high(self):
        out = retry_node(_state(confidence=0.8, answer="Answer"))
        assert out["action"] == "accept"

    def test_accept_at_threshold(self):
        out = retry_node(_state(confidence=DEFAULT_MIN_CONFIDENCE, answer="Answer"))
        assert out["action"] == "accept"

    def test_accept_with_custom_threshold(self):
        out = retry_node(
            _state(confidence=0.3, answer="Answer"),
            min_confidence=0.3,
        )
        assert out["action"] == "accept"

    def test_no_accept_without_answer(self):
        # High confidence but no answer text — should not accept
        out = retry_node(_state(confidence=0.9))
        assert out["action"] != "accept"


# ---------------------------------------------------------------------------
# Retry decision — retry
# ---------------------------------------------------------------------------

class TestRetryAction:
    """Tests for the 'retry' action."""

    def test_retry_when_confidence_low(self):
        out = retry_node(_state(confidence=0.2))
        assert out["action"] == "retry"

    def test_retry_with_zero_confidence(self):
        out = retry_node(_state(confidence=0.0))
        assert out["action"] == "retry"

    def test_retry_below_threshold(self):
        out = retry_node(_state(confidence=DEFAULT_MIN_CONFIDENCE - 0.01))
        assert out["action"] == "retry"

    def test_retry_with_low_answer(self):
        # Has an answer but confidence is below threshold
        out = retry_node(_state(confidence=0.2, answer="Weak answer"))
        assert out["action"] == "retry"

    def test_retry_budget_not_exceeded(self):
        out = retry_node(_state(retry_count=1, confidence=0.1), max_retries=5)
        assert out["action"] == "retry"


# ---------------------------------------------------------------------------
# Retry decision — refuse
# ---------------------------------------------------------------------------

class TestRetryRefuse:
    """Tests for the 'refuse' action."""

    def test_refuse_when_max_retries_reached(self):
        out = retry_node(_state(retry_count=DEFAULT_MAX_RETRIES, confidence=0.1))
        assert out["action"] == "refuse"

    def test_refuse_when_exceeded(self):
        out = retry_node(_state(retry_count=5, confidence=0.1))
        assert out["action"] == "refuse"

    def test_refuse_with_custom_max(self):
        out = retry_node(_state(retry_count=1, confidence=0.1), max_retries=1)
        assert out["action"] == "refuse"

    def test_refuse_on_error(self):
        out = retry_node(_state(error="LLM failed"))
        assert out["action"] == "refuse"

    def test_refuse_on_error_even_if_budget_remains(self):
        out = retry_node(_state(retry_count=0, error="timeout"))
        assert out["action"] == "refuse"

    def test_refuse_on_zero_max_retries(self):
        out = retry_node(_state(confidence=0.1), max_retries=0)
        assert out["action"] == "refuse"


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

class TestRetryCustomParams:
    """Tests for custom max_retries and min_confidence."""

    def test_custom_max_retries(self):
        # retry_count=9, max_retries=10 → still within budget
        out = retry_node(_state(retry_count=9, confidence=0.1), max_retries=10)
        assert out["action"] == "retry"

    def test_custom_min_confidence_accept(self):
        out = retry_node(
            _state(confidence=0.1, answer="Answer"),
            min_confidence=0.1,
        )
        assert out["action"] == "accept"

    def test_custom_min_confidence_retry(self):
        out = retry_node(
            _state(confidence=0.8, answer="Answer"),
            min_confidence=0.9,
        )
        assert out["action"] == "retry"


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestRetryReturnStructure:
    """Verify the shape of returned dict."""

    def test_always_has_retry_count_and_action(self):
        out = retry_node(_state())
        assert "retry_count" in out
        assert "action" in out

    def test_action_is_valid_value(self):
        for conf, ans, err, rc in [
            (0.9, "A", None, 0),
            (0.1, None, None, 0),
            (0.1, None, "err", 0),
            (0.1, None, None, 99),
        ]:
            out = retry_node(_state(
                confidence=conf, answer=ans, error=err, retry_count=rc,
            ))
            assert out["action"] in {"accept", "retry", "refuse"}


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify retry_node is importable from the nodes package."""

    def test_retry_node_exported(self):
        from src.graphs.nodes import retry_node as fn
        assert callable(fn)

    def test_defaults_exported(self):
        from src.graphs.nodes.retry import DEFAULT_MAX_RETRIES, DEFAULT_MIN_CONFIDENCE
        assert isinstance(DEFAULT_MAX_RETRIES, int)
        assert isinstance(DEFAULT_MIN_CONFIDENCE, float)
