"""Tests for the Ollama LLM provider."""

from __future__ import annotations

import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import LLMError
from src.llm import BaseLLM
from src.llm.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    OllamaLLM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_response(text: str = "Generated answer.") -> str:
    """Build a mock Ollama API response JSON string."""
    return json.dumps({
        "model": "llama2",
        "created_at": "2024-01-01T00:00:00Z",
        "response": text,
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 10,
        "eval_count": 5,
    })


def _mock_urlopen(response_data: str | None = None, side_effect: Any | None = None):
    """Create a mock urlopen function."""
    mock_response = MagicMock()
    mock_response.read.return_value = (response_data or _ollama_response()).encode("utf-8")
    mock_response.status = 200
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    
    if side_effect:
        mock_urlopen = MagicMock(side_effect=side_effect)
    else:
        mock_urlopen = MagicMock(return_value=mock_response)
    
    return mock_urlopen


def _http_error(status: int = 404, message: str = "Not Found") -> urllib.error.HTTPError:
    """Build an HTTPError for testing."""
    url = "http://localhost:11434/api/generate"
    return urllib.error.HTTPError(
        url=url,
        code=status,
        msg=message,
        hdrs={},
        fp=None,
    )


def _url_error(reason: str = "Connection refused") -> urllib.error.URLError:
    """Build a URLError for testing."""
    return urllib.error.URLError(reason)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestOllamaLLMConstruction:
    """Tests for OllamaLLM initialisation."""

    def test_default_model(self):
        llm = OllamaLLM()
        assert llm.model_name == DEFAULT_MODEL

    def test_custom_model(self):
        llm = OllamaLLM(model="mistral")
        assert llm.model_name == "mistral"

    def test_custom_base_url(self):
        llm = OllamaLLM(base_url="http://custom:8080")
        assert llm._base_url == "http://custom:8080"

    def test_base_url_trailing_slash_removed(self):
        llm = OllamaLLM(base_url="http://localhost:11434/")
        assert llm._base_url == "http://localhost:11434"

    def test_custom_temperature(self):
        llm = OllamaLLM(temperature=0.7)
        assert llm._temperature == 0.7

    def test_custom_max_tokens(self):
        llm = OllamaLLM(max_tokens=2048)
        assert llm._max_tokens == 2048

    def test_is_base_llm(self):
        llm = OllamaLLM()
        assert isinstance(llm, BaseLLM)


# ---------------------------------------------------------------------------
# Generate method
# ---------------------------------------------------------------------------

class TestOllamaLLMGenerate:
    """Tests for the generate method."""

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_basic(self, mock_urlopen):
        """Test basic generate call."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response("Hello, world!").encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        result = llm.generate("Hello")

        assert result == "Hello, world!"
        mock_urlopen.assert_called_once()

        # Verify the request was made correctly
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.full_url == f"{DEFAULT_BASE_URL}/api/generate"
        assert request.method == "POST"

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_with_custom_model(self, mock_urlopen):
        """Test generate with custom model."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response().encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM(model="mistral")
        llm.generate("Hello")

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        assert body["model"] == "mistral"

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_with_system_prompt(self, mock_urlopen):
        """Test generate with system prompt."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response().encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        llm.generate("Hello", system="You are a helpful assistant.")

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        assert body["system"] == "You are a helpful assistant."

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_with_temperature_override(self, mock_urlopen):
        """Test generate with temperature override."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response().encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM(temperature=0.0)
        llm.generate("Hello", temperature=0.8)

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        assert body["options"]["temperature"] == 0.8

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_with_max_tokens_override(self, mock_urlopen):
        """Test generate with max_tokens override."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response().encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM(max_tokens=1024)
        llm.generate("Hello", max_tokens=512)

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        assert body["options"]["num_predict"] == 512

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_empty_response(self, mock_urlopen):
        """Test generate with empty response."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response("").encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        result = llm.generate("Hello")
        assert result == ""


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestOllamaLLMErrorHandling:
    """Tests for error handling."""

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_model_not_found_error(self, mock_urlopen):
        """Test 404 error for missing model."""
        http_error = _http_error(status=404, message="Not Found")
        http_error.read = MagicMock(return_value=b"model not found")
        mock_urlopen.side_effect = http_error

        llm = OllamaLLM(model="nonexistent")
        with pytest.raises(LLMError) as exc_info:
            llm.generate("Hello")

        assert "not found" in str(exc_info.value).lower()

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_http_500_error(self, mock_urlopen):
        """Test HTTP 500 error."""
        http_error = _http_error(status=500, message="Internal Server Error")
        http_error.read = MagicMock(return_value=b"server error")
        mock_urlopen.side_effect = http_error

        llm = OllamaLLM()
        with pytest.raises(LLMError) as exc_info:
            llm.generate("Hello")

        assert "500" in str(exc_info.value)

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_connection_error_retries(self, mock_urlopen):
        """Test that connection errors are retried."""
        url_error = _url_error("Connection refused")
        
        # Fail twice, then succeed
        mock_response = MagicMock()
        mock_response.read.return_value = _ollama_response().encode("utf-8")
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        
        mock_urlopen.side_effect = [url_error, url_error, mock_response]

        llm = OllamaLLM(max_retries=3, retry_base_delay=0.01)
        result = llm.generate("Hello")

        assert result == "Generated answer."
        assert mock_urlopen.call_count == 3

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_connection_error_exhausted_retries(self, mock_urlopen):
        """Test that connection errors raise after exhausting retries."""
        url_error = _url_error("Connection refused")
        mock_urlopen.side_effect = url_error

        llm = OllamaLLM(max_retries=2, retry_base_delay=0.01)
        with pytest.raises(LLMError) as exc_info:
            llm.generate("Hello")

        assert "2 retries" in str(exc_info.value)
        assert mock_urlopen.call_count == 2

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_invalid_json_response(self, mock_urlopen):
        """Test handling of invalid JSON response."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: b"not valid json",
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        with pytest.raises(LLMError) as exc_info:
            llm.generate("Hello")

        assert "parse" in str(exc_info.value).lower()

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_unexpected_exception(self, mock_urlopen):
        """Test handling of unexpected exceptions."""
        mock_urlopen.side_effect = RuntimeError("Unexpected error")

        llm = OllamaLLM()
        with pytest.raises(LLMError) as exc_info:
            llm.generate("Hello")

        assert "failed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TestOllamaLLMTokenCounting:
    """Tests for token counting."""

    def test_count_tokens_empty_string(self):
        """Test token count for empty string."""
        llm = OllamaLLM()
        assert llm.count_tokens("") == 0

    def test_count_tokens_single_word(self):
        """Test token count for single word."""
        llm = OllamaLLM()
        # 1 word * 1.3 = 1.3 -> int = 1
        assert llm.count_tokens("hello") == 1

    def test_count_tokens_multiple_words(self):
        """Test token count for multiple words."""
        llm = OllamaLLM()
        # 10 words * 1.3 = 13
        text = "The quick brown fox jumps over the lazy dog today"
        assert llm.count_tokens(text) == 13

    def test_count_tokens_returns_int(self):
        """Test that count_tokens returns an integer."""
        llm = OllamaLLM()
        result = llm.count_tokens("some text here")
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestOllamaLLMUtilityMethods:
    """Tests for utility methods."""

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_is_available_true(self, mock_urlopen):
        """Test is_available returns True when server is running."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(status=200)
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        assert llm.is_available() is True

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_is_available_false(self, mock_urlopen):
        """Test is_available returns False when server is not running."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        llm = OllamaLLM()
        assert llm.is_available() is False

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_list_models(self, mock_urlopen):
        """Test list_models returns available models."""
        models_response = json.dumps({
            "models": [
                {"name": "llama2:latest", "size": 4000000000},
                {"name": "mistral:latest", "size": 4000000000},
            ]
        })
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: models_response.encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        models = llm.list_models()

        assert "llama2:latest" in models
        assert "mistral:latest" in models

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_list_models_connection_error(self, mock_urlopen):
        """Test list_models raises LLMError on connection error."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        llm = OllamaLLM()
        with pytest.raises(LLMError):
            llm.list_models()


# ---------------------------------------------------------------------------
# Generate with context
# ---------------------------------------------------------------------------

class TestOllamaLLMGenerateWithContext:
    """Tests for generate_with_context method."""

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_with_context(self, mock_urlopen):
        """Test generate_with_context includes context in prompt."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response("Federalism is...").encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        result = llm.generate_with_context(
            prompt="What is federalism?",
            context=["Federalism is a system of government.", "It divides power between levels."]
        )

        assert result == "Federalism is..."

        # Verify context was included in the prompt
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        prompt = body["prompt"]
        assert "Context:" in prompt
        assert "[1] Federalism is a system of government." in prompt
        assert "[2] It divides power between levels." in prompt
        assert "What is federalism?" in prompt

    @patch("src.llm.ollama.urllib.request.urlopen")
    def test_generate_with_empty_context(self, mock_urlopen):
        """Test generate_with_context with empty context falls back to generate."""
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                read=lambda: _ollama_response("Hello!").encode("utf-8"),
                status=200,
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        llm = OllamaLLM()
        result = llm.generate_with_context(
            prompt="Hello",
            context=[]
        )

        assert result == "Hello!"

        # Verify no context was included
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        prompt = body["prompt"]
        assert "Context:" not in prompt
        assert prompt == "Hello"
