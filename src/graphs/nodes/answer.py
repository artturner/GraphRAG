"""Answer node for generating LLM responses.

This node builds a prompt from retrieved context, calls the LLM, and
returns the generated answer text along with a confidence estimate.

The ``llm`` parameter accepts any object that satisfies the
:class:`BaseLLM` protocol (``generate(prompt, **kwargs) -> str`` and
a ``model_name`` property).  The concrete ``BaseLLM`` ABC is defined
in ``src.llm.base`` (Chunk 8).
"""

import logging
from typing import Any, Protocol, runtime_checkable

from src.graphs.prompts import build_rag_prompt, build_refusal_prompt
from src.graphs.state import GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocol so this module works before src.llm exists
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseLLM(Protocol):
    """Minimal LLM interface required by the answer node."""

    @property
    def model_name(self) -> str: ...  # pragma: no cover

    def generate(self, prompt: str, **kwargs: Any) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def answer_node(state: GraphState, llm: BaseLLM) -> dict:
    """Generate an answer using the LLM and context from *state*.

    Behaviour:
    1. If ``state["search_results"]`` is empty, build a refusal prompt
       and set ``refusal_reason`` instead of ``answer``.
    2. Otherwise, assemble the context from the chunk texts, build a
       RAG prompt, call ``llm.generate``, and return the answer.
    3. If the LLM call raises an exception, catch it and set ``error``.

    Args:
        state: Current graph state — should contain ``question`` and
            ``search_results`` (populated by the retrieve node).
        llm: An object implementing the :class:`BaseLLM` protocol.

    Returns:
        A dict with ``answer`` and ``confidence`` on success, or
        ``refusal_reason`` when context is empty, or ``error`` on
        failure.

    Example:
        ```python
        result = answer_node(
            {"question": "What is federalism?", "search_results": [...]},
            llm,
        )
        # result == {"answer": "Federalism is...", "confidence": 0.85}
        ```
    """
    question: str = state.get("question", "")

    if not question or not question.strip():
        logger.warning("answer_node received empty question")
        return {"answer": None, "confidence": 0.0, "error": "No question provided"}

    search_results = state.get("search_results", [])

    # ------------------------------------------------------------------
    # No context → refuse
    # ------------------------------------------------------------------
    if not search_results:
        logger.info("No search results — generating refusal")
        try:
            prompt = build_refusal_prompt(question)
            refusal_text = llm.generate(prompt)
            return {
                "answer": None,
                "confidence": 0.0,
                "refusal_reason": refusal_text.strip(),
            }
        except Exception as e:
            logger.error("LLM refusal generation failed: %s", e)
            return {
                "answer": None,
                "confidence": 0.0,
                "refusal_reason": "Insufficient context to answer the question.",
            }

    # ------------------------------------------------------------------
    # Build prompt and generate
    # ------------------------------------------------------------------
    context_texts = [r.chunk.content for r in search_results]

    try:
        prompt = build_rag_prompt(question, context_texts)
        answer_text = llm.generate(prompt)

        # Derive a simple confidence from the average search score
        scores = [r.score for r in search_results]
        confidence = sum(scores) / len(scores) if scores else 0.0

        logger.debug(
            "Generated answer (confidence=%.2f) for: %s",
            confidence,
            question[:80],
        )

        return {
            "answer": answer_text.strip(),
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return {
            "answer": None,
            "confidence": 0.0,
            "error": f"LLM generation failed: {e}",
        }
