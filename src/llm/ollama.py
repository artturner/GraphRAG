"""Ollama LLM provider for local models.

This module implements :class:`OllamaLLM`, a concrete :class:`BaseLLM`
subclass that calls Ollama's local API for running open-source models
like Llama 2, Mistral, and others.

Usage::

    from src.llm.ollama import OllamaLLM

    llm = OllamaLLM(model="llama2")
    response = llm.generate("What is federalism?")
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from src.exceptions import LLMError
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama2"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 120  # Ollama can be slow for larger models
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0


class OllamaLLM(BaseLLM):
    """LLM provider that calls Ollama's local API.

    Ollama runs models locally via a REST API. This provider supports
    any model available in Ollama (llama2, mistral, codellama, etc.).

    Args:
        model: Ollama model name (e.g. ``"llama2"``, ``"mistral"``).
        base_url: Ollama API base URL. Defaults to ``"http://localhost:11434"``.
        temperature: Sampling temperature. Defaults to ``0.0``.
        max_tokens: Maximum tokens in the response. Defaults to ``1024``.
        timeout: Request timeout in seconds. Defaults to ``120``.
        max_retries: Number of retry attempts for connection errors.
        retry_base_delay: Base delay (seconds) for exponential back-off.
        client: Optional pre-configured client for testing.
            Useful for dependency injection.

    Example:
        ```python
        llm = OllamaLLM(model="llama2")
        print(llm.generate("Explain federalism."))
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = client

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response via the Ollama API.

        Args:
            prompt: The user message.
            **kwargs: Overrides for ``temperature``, ``max_tokens``, or
                ``system`` (system prompt string).

        Returns:
            The generated text.

        Raises:
            LLMError: On connection errors or after exhausting all retry attempts.
        """
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        system_prompt: str | None = kwargs.get("system")

        # Build the request payload for Ollama's generate endpoint
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        return self._call_with_retry(payload)

    def count_tokens(self, text: str) -> int:
        """Estimate token count using word-based approximation.

        Ollama models use different tokenizers depending on the model.
        This implementation uses a simple word-count approximation that
        works reasonably well for most models.

        Args:
            text: The string to tokenise.

        Returns:
            Estimated token count (non-negative).
        """
        if not text:
            return 0
        # Approximate: ~1.3 tokens per word on average for English text
        # This is a rough estimate that works across different tokenizers
        words = len(text.split())
        return int(words * 1.3)

    # ------------------------------------------------------------------
    # API communication
    # ------------------------------------------------------------------

    def _call_with_retry(self, payload: dict[str, Any]) -> str:
        """Call the Ollama API with exponential back-off for connection errors."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return self._make_request(payload)

            except urllib.error.URLError as exc:
                last_error = exc
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama connection error (attempt %d/%d), sleeping %.1fs: %s",
                    attempt,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

            except LLMError:
                # Non-retryable LLM errors should propagate immediately
                raise

            except Exception as exc:
                raise LLMError(
                    "Ollama call failed",
                    details=str(exc),
                ) from exc

        # All retries exhausted
        raise LLMError(
            f"Ollama call failed after {self._max_retries} retries",
            details=str(last_error),
        )

    def _make_request(self, payload: dict[str, Any]) -> str:
        """Make a single HTTP request to the Ollama API."""
        url = f"{self._base_url}/api/generate"
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                response_data = response.read().decode("utf-8")
                return self._extract_text(response_data)

        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                pass

            if exc.code == 404:
                raise LLMError(
                    f"Model '{self._model}' not found in Ollama",
                    details=f"Run 'ollama pull {self._model}' to download the model.",
                ) from exc

            raise LLMError(
                f"Ollama API error: HTTP {exc.code}",
                details=error_body or str(exc),
            ) from exc

        except urllib.error.URLError as exc:
            # Connection refused, timeout, etc. - retryable
            raise

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response_data: str) -> str:
        """Extract the generated text from an Ollama API response."""
        try:
            data = json.loads(response_data)
            # Ollama's generate endpoint returns a 'response' field
            return data.get("response", "")
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Failed to parse Ollama response",
                details=f"Invalid JSON: {exc}",
            ) from exc
        except (TypeError, AttributeError) as exc:
            raise LLMError(
                "Failed to parse Ollama response",
                details=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the Ollama server is running and accessible.

        Returns:
            True if the Ollama server is responding, False otherwise.
        """
        try:
            url = f"{self._base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models in the Ollama server.

        Returns:
            List of model names.

        Raises:
            LLMError: If unable to connect to Ollama server.
        """
        try:
            url = f"{self._base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = data.get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
        except urllib.error.URLError as exc:
            raise LLMError(
                "Failed to connect to Ollama server",
                details=str(exc),
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Failed to parse Ollama response",
                details=str(exc),
            ) from exc
