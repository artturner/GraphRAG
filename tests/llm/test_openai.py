"""Tests for the OpenAI LLM provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from src.exceptions import LLMError
from src.llm import BaseLLM
from src.llm.openai_llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    OpenAILLM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_response(text: str = "Generated answer.") -> SimpleNamespace:
    """Build a mock Chat Completions response object."""
    message = SimpleNamespace(role="assistant", content=text)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(
        id="chatcmpl-abc",
        choices=[choice],
        usage=usage,
        model="gpt-4-turbo-preview",
    )


def _mock_client(response: Any | None = None) -> MagicMock:
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.chat.completions.create.return_value = response or _chat_response()
    return client


def _rate_limit_error() -> RateLimitError:
    """Build a RateLimitError."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "rate limited"}}
    return RateLimitError(
        message="rate limited",
        response=mock_response,
        body={"error": {"message": "rate limited"}},
    )


def _api_status_error(status: int = 400, message: str = "bad request") -> APIStatusError:
    """Build an APIStatusError."""
    mock_response = MagicMock()
    mock_response.status_code = status
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": message}}
    return APIStatusError(
        message=message,
        response=mock_response,
        body={"error": {"message": message}},
    )


def _timeout_error() -> APITimeoutError:
    """Build an APITimeoutError."""
    request = MagicMock()
    return APITimeoutError(request=request)


def _connection_error() -> APIConnectionError:
    """Build an APIConnectionError."""
    request = MagicMock()
    return APIConnectionError(request=request)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestOpenAILLMConstruction:
    """Tests for OpenAILLM initialisation."""

    def test_default_model(self):
        llm = OpenAILLM(client=_mock_client())
        assert llm.model_name == DEFAULT_MODEL

    def test_custom_model(self):
        llm = OpenAILLM(model="gpt-3.5-turbo", client=_mock_client())
        assert llm.model_name == "gpt-3.5-turbo"

    def test_is_base_llm(self):
        llm = OpenAILLM(client=_mock_client())
        assert isinstance(llm, BaseLLM)

    def test_satisfies_graph_protocol(self):
        from src.graphs.nodes.answer import BaseLLM as LLMProtocol

        llm = OpenAILLM(client=_mock_client())
        assert isinstance(llm, LLMProtocol)

    def test_lazy_client_creation(self):
        """Client should NOT be created until first API call."""
        with patch("src.llm.openai_llm.OpenAI") as mock_cls:
            llm = OpenAILLM(api_key="sk-test")
            mock_cls.assert_not_called()

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY if present
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            llm = OpenAILLM()
            with pytest.raises(LLMError, match="API key"):
                llm.generate("prompt")


# ---------------------------------------------------------------------------
# generate — success
# ---------------------------------------------------------------------------

class TestOpenAIGenerate:
    """Tests for the generate method."""

    def test_returns_text(self):
        client = _mock_client(_chat_response("Federalism is a system."))
        llm = OpenAILLM(client=client)

        result = llm.generate("What is federalism?")

        assert result == "Federalism is a system."

    def test_passes_model(self):
        client = _mock_client()
        llm = OpenAILLM(model="gpt-4", client=client)

        llm.generate("prompt")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4"

    def test_passes_default_temperature(self):
        client = _mock_client()
        llm = OpenAILLM(client=client)

        llm.generate("prompt")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == DEFAULT_TEMPERATURE

    def test_passes_default_max_tokens(self):
        client = _mock_client()
        llm = OpenAILLM(client=client)

        llm.generate("prompt")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_custom_temperature(self):
        client = _mock_client()
        llm = OpenAILLM(temperature=0.7, client=client)

        llm.generate("prompt")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_kwarg_temperature_overrides(self):
        client = _mock_client()
        llm = OpenAILLM(temperature=0.0, client=client)

        llm.generate("prompt", temperature=0.9)

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_kwarg_max_tokens_overrides(self):
        client = _mock_client()
        llm = OpenAILLM(max_tokens=100, client=client)

        llm.generate("prompt", max_tokens=500)

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 500

    def test_system_prompt(self):
        client = _mock_client()
        llm = OpenAILLM(client=client)

        llm.generate("prompt", system="You are a helpful tutor.")

        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a helpful tutor."}
        assert messages[1] == {"role": "user", "content": "prompt"}

    def test_no_system_prompt_by_default(self):
        client = _mock_client()
        llm = OpenAILLM(client=client)

        llm.generate("prompt")

        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_user_message_content(self):
        client = _mock_client()
        llm = OpenAILLM(client=client)

        llm.generate("What is X?")

        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "user", "content": "What is X?"}

    def test_none_content_returns_empty(self):
        """Handle response where content is None."""
        message = SimpleNamespace(role="assistant", content=None)
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice])
        client = _mock_client(response)
        llm = OpenAILLM(client=client)

        result = llm.generate("prompt")

        assert result == ""


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestOpenAICountTokens:
    """Tests for tiktoken-based token counting."""

    def test_empty_string(self):
        llm = OpenAILLM(client=_mock_client())
        assert llm.count_tokens("") == 0

    def test_short_text(self):
        llm = OpenAILLM(client=_mock_client())
        count = llm.count_tokens("hello")
        assert count >= 1

    def test_returns_int(self):
        llm = OpenAILLM(client=_mock_client())
        assert isinstance(llm.count_tokens("some text here"), int)

    def test_longer_text_more_tokens(self):
        llm = OpenAILLM(client=_mock_client())
        short = llm.count_tokens("hi")
        long = llm.count_tokens("This is a much longer sentence with many words.")
        assert long > short

    def test_fallback_encoding_for_unknown_model(self):
        """Unknown models should fall back to cl100k_base."""
        llm = OpenAILLM(model="unknown-model-xyz", client=_mock_client())
        count = llm.count_tokens("hello world")
        assert count >= 1

    def test_consistent_results(self):
        llm = OpenAILLM(client=_mock_client())
        text = "The quick brown fox jumps over the lazy dog."
        assert llm.count_tokens(text) == llm.count_tokens(text)


# ---------------------------------------------------------------------------
# generate_with_context (inherited)
# ---------------------------------------------------------------------------

class TestOpenAIGenerateWithContext:
    """Tests for the inherited generate_with_context method."""

    def test_empty_context(self):
        client = _mock_client(_chat_response("Direct answer."))
        llm = OpenAILLM(client=client)

        result = llm.generate_with_context("Q?", [])

        assert result == "Direct answer."

    def test_with_context(self):
        client = _mock_client(_chat_response("Contextual answer."))
        llm = OpenAILLM(client=client)

        llm.generate_with_context("Q?", ["passage one", "passage two"])

        call_kwargs = client.chat.completions.create.call_args[1]
        prompt_text = call_kwargs["messages"][-1]["content"]
        assert "[1] passage one" in prompt_text
        assert "[2] passage two" in prompt_text


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestOpenAIErrorHandling:
    """Tests for error handling and retries."""

    def test_non_retryable_status_error_raises_immediately(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = _api_status_error(400, "bad request")
        llm = OpenAILLM(client=client, max_retries=3, retry_base_delay=0)

        with pytest.raises(LLMError, match="400"):
            llm.generate("prompt")

        assert client.chat.completions.create.call_count == 1

    def test_auth_error_raises_immediately(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = _api_status_error(401, "unauthorized")
        llm = OpenAILLM(client=client, max_retries=3, retry_base_delay=0)

        with pytest.raises(LLMError, match="401"):
            llm.generate("prompt")

        assert client.chat.completions.create.call_count == 1

    def test_rate_limit_retries(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _rate_limit_error(),
            _chat_response("Success after retries."),
        ]
        llm = OpenAILLM(client=client, max_retries=3, retry_base_delay=0)

        result = llm.generate("prompt")

        assert result == "Success after retries."
        assert client.chat.completions.create.call_count == 3

    def test_timeout_retries(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = [
            _timeout_error(),
            _chat_response("Recovered."),
        ]
        llm = OpenAILLM(client=client, max_retries=3, retry_base_delay=0)

        result = llm.generate("prompt")

        assert result == "Recovered."

    def test_connection_error_retries(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = [
            _connection_error(),
            _chat_response("Back online."),
        ]
        llm = OpenAILLM(client=client, max_retries=3, retry_base_delay=0)

        result = llm.generate("prompt")

        assert result == "Back online."

    def test_retries_exhausted_raises(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = _rate_limit_error()
        llm = OpenAILLM(client=client, max_retries=2, retry_base_delay=0)

        with pytest.raises(LLMError, match="after 2 retries"):
            llm.generate("prompt")

        assert client.chat.completions.create.call_count == 2

    def test_generic_exception_raises_llm_error(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = RuntimeError("connection reset")
        llm = OpenAILLM(client=client, retry_base_delay=0)

        with pytest.raises(LLMError, match="OpenAI call failed"):
            llm.generate("prompt")

    def test_malformed_response_raises(self):
        response = SimpleNamespace(choices=[])
        client = _mock_client(response)
        llm = OpenAILLM(client=client)

        with pytest.raises(LLMError, match="parse"):
            llm.generate("prompt")

    def test_llm_error_has_details(self):
        client = _mock_client()
        client.chat.completions.create.side_effect = _api_status_error(
            404, "model not found"
        )
        llm = OpenAILLM(client=client, retry_base_delay=0)

        with pytest.raises(LLMError) as exc_info:
            llm.generate("prompt")

        assert exc_info.value.details is not None


# ---------------------------------------------------------------------------
# Lazy client creation
# ---------------------------------------------------------------------------

class TestOpenAIClientCreation:
    """Tests for lazy client creation."""

    def test_injected_client_used(self):
        client = _mock_client(_chat_response("ok"))
        llm = OpenAILLM(client=client)

        llm.generate("prompt")

        client.chat.completions.create.assert_called_once()

    def test_creates_client_on_first_call(self):
        with patch("src.llm.openai_llm.OpenAI") as mock_cls:
            mock_instance = _mock_client(_chat_response("ok"))
            mock_cls.return_value = mock_instance

            llm = OpenAILLM(api_key="sk-test123")
            mock_cls.assert_not_called()

            llm.generate("prompt")

            mock_cls.assert_called_once_with(api_key="sk-test123")

    def test_client_reused_across_calls(self):
        with patch("src.llm.openai_llm.OpenAI") as mock_cls:
            mock_instance = _mock_client(_chat_response("ok"))
            mock_cls.return_value = mock_instance

            llm = OpenAILLM(api_key="sk-test")
            llm.generate("first")
            llm.generate("second")

            mock_cls.assert_called_once()
            assert mock_instance.chat.completions.create.call_count == 2

    def test_env_var_api_key(self):
        with patch("src.llm.openai_llm.OpenAI") as mock_cls, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            mock_instance = _mock_client(_chat_response("ok"))
            mock_cls.return_value = mock_instance

            llm = OpenAILLM()  # no explicit api_key
            llm.generate("prompt")

            mock_cls.assert_called_once_with(api_key="sk-env")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify OpenAILLM is importable."""

    def test_importable_from_module(self):
        from src.llm.openai_llm import OpenAILLM as cls
        assert cls is not None

    def test_defaults_exported(self):
        from src.llm.openai_llm import (
            DEFAULT_MAX_TOKENS,
            DEFAULT_MODEL,
            DEFAULT_TEMPERATURE,
        )
        assert isinstance(DEFAULT_MODEL, str)
        assert isinstance(DEFAULT_TEMPERATURE, float)
        assert isinstance(DEFAULT_MAX_TOKENS, int)
