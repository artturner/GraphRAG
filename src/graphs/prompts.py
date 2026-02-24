"""Prompt templates and builder utilities for the Q&A workflow.

This module centralises all prompt strings so that they can be tuned,
tested, and swapped independently of the graph nodes that use them.
"""


# ---------------------------------------------------------------------------
# Core prompt templates
# ---------------------------------------------------------------------------

RAG_PROMPT = """\
Use the following context to answer the question. \
Only use information from the provided context. \
If the context does not contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""


REFUSAL_PROMPT = """\
The system was unable to find sufficient evidence to answer the \
following question. Provide a polite refusal explaining that the \
available documents do not contain enough information.

Question: {question}

Refusal:"""


# ---------------------------------------------------------------------------
# Builder utilities
# ---------------------------------------------------------------------------


def build_rag_prompt(question: str, context_texts: list[str]) -> str:
    """Build a RAG prompt from a question and a list of context passages.

    Each context passage is separated by a blank line and prefixed with
    a 1-based index for easy reference.

    Args:
        question: The user's question.
        context_texts: List of text passages retrieved from the corpus.

    Returns:
        The formatted prompt string ready for LLM generation.

    Example:
        ```python
        prompt = build_rag_prompt(
            "What is federalism?",
            ["Federalism is a system...", "In a federal system..."],
        )
        ```
    """
    if not context_texts:
        context_block = "(No relevant context was found.)"
    else:
        parts = [
            f"[{i}] {text}" for i, text in enumerate(context_texts, 1)
        ]
        context_block = "\n\n".join(parts)

    return RAG_PROMPT.format(context=context_block, question=question)


def build_refusal_prompt(question: str) -> str:
    """Build a refusal prompt for an unanswerable question.

    Args:
        question: The user's question.

    Returns:
        The formatted refusal prompt string.
    """
    return REFUSAL_PROMPT.format(question=question)


SYNTHESIS_PROMPT = """\
You are a knowledgeable research assistant. Use the following context passages \
as your primary source material. You may draw connections, identify patterns, \
and propose ideas that go beyond the literal text of the passages, provided \
your response is grounded in and consistent with the retrieved material.

Context:
{context}

Request: {question}

Response:"""


def build_synthesis_prompt(question: str, context_texts: list[str]) -> str:
    """Build a synthesis prompt from a question and a list of context passages.

    Args:
        question: The user's synthesis request.
        context_texts: List of text passages retrieved from the corpus.

    Returns:
        The formatted prompt string ready for LLM generation.
    """
    if not context_texts:
        context_block = "(No relevant context was found.)"
    else:
        parts = [f"[{i}] {text}" for i, text in enumerate(context_texts, 1)]
        context_block = "\n\n".join(parts)
    return SYNTHESIS_PROMPT.format(context=context_block, question=question)
