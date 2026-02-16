"""Grounding-verification node for the LangGraph Q&A workflow.

This node checks whether the generated answer is supported by the
retrieved chunks.  It delegates the heavy lifting to
:class:`~src.graphs.grounding.GroundingChecker` and writes the
results back into the graph state.
"""

import logging

from src.graphs.grounding import GroundingChecker
from src.graphs.state import GraphState

logger = logging.getLogger(__name__)

# Module-level default so callers don't have to construct one
_DEFAULT_CHECKER = GroundingChecker()


def verify_grounding_node(
    state: GraphState,
    checker: GroundingChecker | None = None,
) -> dict:
    """Verify whether the answer in *state* is grounded in its chunks.

    Args:
        state: Current graph state — should already contain ``answer``
            and ``chunks`` (populated by the retrieve and answer nodes).
        checker: An optional :class:`GroundingChecker` instance.  When
            ``None`` a module-level default (threshold=0.3,
            min_confidence=0.5) is used.

    Returns:
        A dict with ``confidence``, ``is_grounded``, and optionally
        ``unsupported_claims`` or ``error``.

    Example:
        ```python
        result = verify_grounding_node({
            "answer": "Federalism divides power...",
            "chunks": [Chunk(content="Federalism divides power...")],
        })
        # {"confidence": 1.0, "is_grounded": True}
        ```
    """
    checker = checker or _DEFAULT_CHECKER
    answer: str | None = state.get("answer")
    chunks = state.get("chunks", [])

    # -- guard: no answer to verify ------------------------------------
    if not answer or not answer.strip():
        logger.warning("verify_grounding_node: no answer to verify")
        return {
            "confidence": 0.0,
            "is_grounded": False,
            "error": "No answer to verify",
        }

    # -- run the grounding check ---------------------------------------
    try:
        result = checker.check_grounding(answer, chunks)

        out: dict = {
            "confidence": result.confidence,
            "is_grounded": result.is_grounded,
        }

        if result.unsupported_claims:
            out["unsupported_claims"] = result.unsupported_claims

        logger.debug(
            "Grounding check: grounded=%s confidence=%.2f (%d unsupported)",
            result.is_grounded,
            result.confidence,
            len(result.unsupported_claims),
        )

        return out

    except Exception as e:
        logger.error("Grounding verification failed: %s", e)
        return {
            "confidence": 0.0,
            "is_grounded": False,
            "error": f"Grounding verification failed: {e}",
        }
