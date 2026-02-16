"""AWS Bedrock LLM provider for Claude models.

This module implements :class:`BedrockLLM`, a concrete :class:`BaseLLM`
subclass that calls Anthropic Claude models via the Amazon Bedrock
``converse`` API.

Usage::

    from src.llm.bedrock import BedrockLLM

    llm = BedrockLLM(model="anthropic.claude-3-sonnet-20240229-v1:0")
    response = llm.generate("What is federalism?")
"""

from __future__ import annotations

import logging
import time
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from src.exceptions import LLMError
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0

# Approximate characters-per-token ratio for Claude
_CHARS_PER_TOKEN = 4


class BedrockLLM(BaseLLM):
    """LLM provider that calls Anthropic Claude via AWS Bedrock.

    The provider uses the Bedrock **Converse** API which provides a
    unified interface across model families.  Retries with exponential
    back-off are applied for transient ``ThrottlingException`` and
    ``ServiceUnavailableException`` errors.

    Args:
        model: Bedrock model ID
            (e.g. ``"anthropic.claude-3-sonnet-20240229-v1:0"``).
        region: AWS region name.  Defaults to ``"us-east-1"``.
        temperature: Sampling temperature.  Defaults to ``0.0``.
        max_tokens: Maximum tokens in the response.  Defaults to ``1024``.
        max_retries: Number of retry attempts for transient errors.
        retry_base_delay: Base delay (seconds) for exponential back-off.
        client: Optional pre-configured ``boto3`` Bedrock-runtime client.
            Useful for testing and dependency injection.

    Example:
        ```python
        llm = BedrockLLM(model="anthropic.claude-3-sonnet-20240229-v1:0")
        print(llm.generate("Explain federalism."))
        ```
    """

    # Error codes that warrant a retry
    _RETRYABLE_ERRORS: frozenset[str] = frozenset({
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
    })

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        region: str = DEFAULT_REGION,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._region = region
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = client

    # ------------------------------------------------------------------
    # Lazy client creation
    # ------------------------------------------------------------------

    @property
    def _bedrock(self) -> Any:
        """Return the Bedrock-runtime client, creating it on first use."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=BotoConfig(
                    retries={"max_attempts": 0},  # we handle retries ourselves
                ),
            )
        return self._client

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the Bedrock Converse API.

        Args:
            prompt: The user prompt.
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

        messages = [
            {"role": "user", "content": [{"text": prompt}]},
        ]

        inference_config: dict[str, Any] = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        api_kwargs: dict[str, Any] = {
            "modelId": self._model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if system_prompt:
            api_kwargs["system"] = [{"text": system_prompt}]

        return self._call_with_retry(api_kwargs)

    def count_tokens(self, text: str) -> int:
        """Estimate token count using a character-based heuristic.

        Claude's tokeniser is not publicly available outside the API,
        so we approximate with ~4 characters per token.

        Args:
            text: The string to estimate.

        Returns:
            Estimated token count (non-negative).
        """
        if not text:
            return 0
        return max(1, len(text) // _CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _call_with_retry(self, api_kwargs: dict[str, Any]) -> str:
        """Call the Converse API with exponential-back-off retry."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._bedrock.converse(**api_kwargs)
                return self._extract_text(response)

            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")

                if error_code not in self._RETRYABLE_ERRORS:
                    raise LLMError(
                        f"Bedrock API error: {error_code}",
                        details=str(exc),
                    ) from exc

                last_error = exc
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable error %s (attempt %d/%d), sleeping %.1fs",
                    error_code,
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)

            except Exception as exc:
                raise LLMError(
                    "Bedrock call failed",
                    details=str(exc),
                ) from exc

        # All retries exhausted
        raise LLMError(
            f"Bedrock call failed after {self._max_retries} retries",
            details=str(last_error),
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Extract the assistant's text from a Converse API response."""
        try:
            output = response["output"]["message"]["content"]
            parts = [block["text"] for block in output if "text" in block]
            return "".join(parts)
        except (KeyError, TypeError, IndexError) as exc:
            raise LLMError(
                "Failed to parse Bedrock response",
                details=str(exc),
            ) from exc
