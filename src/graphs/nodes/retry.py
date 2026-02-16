"""Retry node for the LangGraph Q&A workflow.

This node inspects the current state (confidence, retry count) and
decides whether the workflow should loop back for another attempt or
give up and proceed to refusal.
"""

import logging

from src.graphs.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_MIN_CONFIDENCE: float = 0.5


def retry_node(
    state: GraphState,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict:
    """Decide whether to retry answer generation or refuse.

    The node increments ``retry_count`` and returns an ``action`` field
    that downstream routing can use:

    * ``"retry"`` – confidence is below *min_confidence* and the retry
      budget has not been exhausted.
    * ``"refuse"`` – retries are exhausted **or** the state already
      carries an unrecoverable error.
    * ``"accept"`` – the answer meets the confidence threshold.

    Args:
        state: Current graph state.
        max_retries: Maximum number of retries allowed (default 3).
        min_confidence: Confidence threshold below which a retry is
            triggered (default 0.5).

    Returns:
        A dict with ``retry_count`` and ``action``.
    """
    retry_count: int = state.get("retry_count", 0)
    confidence: float = state.get("confidence", 0.0)
    error: str | None = state.get("error")
    answer: str | None = state.get("answer")

    # Always bump the counter so the graph converges
    new_count = retry_count + 1

    # -- accept: answer is good enough ---------------------------------
    if answer and confidence >= min_confidence:
        logger.debug(
            "Accepting answer (confidence=%.2f >= %.2f)",
            confidence,
            min_confidence,
        )
        return {"retry_count": new_count, "action": "accept"}

    # -- refuse: budget exhausted or hard error ------------------------
    if new_count > max_retries:
        logger.info(
            "Retry budget exhausted (%d/%d) — refusing",
            new_count,
            max_retries,
        )
        return {"retry_count": new_count, "action": "refuse"}

    if error:
        logger.info("Unrecoverable error detected — refusing: %s", error)
        return {"retry_count": new_count, "action": "refuse"}

    # -- retry: try again ----------------------------------------------
    logger.info(
        "Retrying (%d/%d) — confidence %.2f < %.2f",
        new_count,
        max_retries,
        confidence,
        min_confidence,
    )
    return {"retry_count": new_count, "action": "retry"}
