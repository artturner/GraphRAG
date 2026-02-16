"""OpenAI LLM provider for GPT models.

This module implements :class:`OpenAILLM`, a concrete :class:`BaseLLM`
subclass that calls OpenAI's Chat Completions API via the official
``openai`` Python library (v2+).

Usage::

    from src.llm.openai_llm import OpenAILLM

    llm = OpenAILLM(model="gpt-4-turbo-preview")
    response = llm.generate("What is federalism?")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import tiktoken
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.exceptions import LLMError
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4-turbo-preview"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0

# Fallback tiktoken encoding when model-specific encoding is unavailable
_FALLBACK_ENCODING = "cl100k_base"


class OpenAILLM(BaseLLM):
    """LLM provider that calls OpenAI's Chat Completions API.

    The provider supports GPT-4, GPT-4-turbo, and GPT-3.5-turbo model
    families.  Retries with exponential back-off are applied for
    transient errors (rate limits, timeouts, connection failures).

    Args:
        model: OpenAI model name (e.g. ``"gpt-4-turbo-preview"``).
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when ``None``.
        temperature: Sampling temperature.  Defaults to ``0.0``.
        max_tokens: Maximum tokens in the response.  Defaults to ``1024``.
        max_retries: Number of retry attempts for transient errors.
        retry_base_delay: Base delay (seconds) for exponential back-off.
        client: Optional pre-configured :class:`openai.OpenAI` client.
            Useful for testing and dependency injection.

    Example:
        ```python
        llm = OpenAILLM(model="gpt-4-turbo-preview")
        print(llm.generate("Explain federalism."))
        ```
    """

    # Error types that warrant a retry
    _RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = client
        self._encoding: tiktoken.Encoding | None = None

    # ------------------------------------------------------------------
    # Lazy client creation
    # ------------------------------------------------------------------

    @property
    def _openai(self) -> Any:
        """Return the OpenAI client, creating it on first use."""
        if self._client is None:
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMError(
                    "OpenAI API key not provided",
                    details="Set OPENAI_API_KEY or pass api_key=",
                )
            self._client = OpenAI(api_key=key)
        return self._client

    # ------------------------------------------------------------------
    # Tiktoken encoding
    # ------------------------------------------------------------------

    def _get_encoding(self) -> tiktoken.Encoding:
        """Return the tiktoken encoding for the current model."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)
        return self._encoding

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response via the Chat Completions API.

        Args:
            prompt: The user message.
            **kwargs: Overrides for ``temperature``, ``max_tokens``, or
                ``system`` (system prompt string).

        Returns:
            The generated text.

        Raises:
            LLMError: On non-retryable API errors or after exhausting
                all retry attempts.
        """
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        system_prompt: str | None = kwargs.get("system")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        return self._call_with_retry(api_kwargs)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tiktoken encoding.

        Args:
            text: The string to tokenise.

        Returns:
            Exact token count (non-negative).
        """
        if not text:
            return 0
        return len(self._get_encoding().encode(text))

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _call_with_retry(self, api_kwargs: dict[str, Any]) -> str:
        """Call the Chat Completions API with exponential back-off."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._openai.chat.completions.create(**api_kwargs)
                return self._extract_text(response)

            except self._RETRYABLE_ERRORS as exc:
                last_error = exc
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable error %s (attempt %d/%d), sleeping %.1fs",
                    type(exc).__name__,
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)

            except APIStatusError as exc:
                raise LLMError(
                    f"OpenAI API error: {exc.status_code}",
                    details=str(exc),
                ) from exc

            except Exception as exc:
                raise LLMError(
                    "OpenAI call failed",
                    details=str(exc),
                ) from exc

        # All retries exhausted
        raise LLMError(
            f"OpenAI call failed after {self._max_retries} retries",
            details=str(last_error),
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the assistant message text from a completions response."""
        try:
            choice = response.choices[0]
            return choice.message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(
                "Failed to parse OpenAI response",
                details=str(exc),
            ) from exc
