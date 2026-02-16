"""Abstract base class for LLM providers.

Every LLM backend (OpenAI, Bedrock, Ollama, …) must subclass
:class:`BaseLLM` and implement its abstract methods.  The interface is
intentionally kept small so that new providers can be added with
minimal boilerplate.

The class is also compatible with the lightweight
:class:`~src.graphs.nodes.answer.BaseLLM` *Protocol* used by the graph
nodes — any concrete ``BaseLLM`` subclass automatically satisfies that
protocol.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class that every LLM provider must implement.

    Subclasses **must** provide:

    * :pyattr:`model_name` — a read-only property returning the model
      identifier (e.g. ``"gpt-4"``).
    * :pymeth:`generate` — send a prompt and return the generated text.
    * :pymeth:`count_tokens` — estimate the token count for a string.

    The :pymeth:`generate_with_context` method has a default
    implementation that concatenates context passages and delegates to
    :pymeth:`generate`, but providers may override it for more
    efficient context handling.

    Example:
        ```python
        class MyLLM(BaseLLM):
            @property
            def model_name(self) -> str:
                return "my-model"

            def generate(self, prompt: str, **kwargs) -> str:
                return call_my_backend(prompt)

            def count_tokens(self, text: str) -> int:
                return len(text.split())
        ```
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the underlying model."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text completion for *prompt*.

        Args:
            prompt: The input prompt string.
            **kwargs: Provider-specific options (temperature, max_tokens,
                stop sequences, etc.).

        Returns:
            The generated text.

        Raises:
            RuntimeError: If the backend call fails.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the estimated number of tokens in *text*.

        Providers should use their own tokeniser when available;
        a word-count approximation is acceptable as a fallback.

        Args:
            text: The string to tokenise.

        Returns:
            A non-negative integer token count.
        """

    # ------------------------------------------------------------------
    # Default implementation
    # ------------------------------------------------------------------

    def generate_with_context(
        self,
        prompt: str,
        context: list[str],
        **kwargs: Any,
    ) -> str:
        """Generate a response using *prompt* augmented with *context*.

        The default implementation numbers each context passage and
        prepends them to the prompt, then delegates to :pymeth:`generate`.
        Providers that support native context windows or tool-use may
        override this for better efficiency.

        Args:
            prompt: The user prompt / question.
            context: A list of text passages to include as context.
            **kwargs: Forwarded to :pymeth:`generate`.

        Returns:
            The generated text.
        """
        if not context:
            return self.generate(prompt, **kwargs)

        parts = [f"[{i}] {text}" for i, text in enumerate(context, 1)]
        context_block = "\n\n".join(parts)

        full_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer:"
        )

        return self.generate(full_prompt, **kwargs)
