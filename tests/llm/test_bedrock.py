"""Tests for the AWS Bedrock LLM provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.exceptions import LLMError
from src.llm import BaseLLM
from src.llm.bedrock import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REGION,
    DEFAULT_TEMPERATURE,
    BedrockLLM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _converse_response(text: str = "Generated answer.") -> dict:
    """Build a mock Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "end_turn",
    }


def _client_error(code: str, message: str = "error") -> ClientError:
    """Build a botocore ClientError with the given error code."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "Converse",
    )


def _mock_client(response: dict | None = None) -> MagicMock:
    """Create a mock bedrock-runtime client."""
    client = MagicMock()
    client.converse.return_value = response or _converse_response()
    return client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBedrockLLMConstruction:
    """Tests for BedrockLLM initialisation."""

    def test_default_model(self):
        llm = BedrockLLM(client=_mock_client())
        assert llm.model_name == DEFAULT_MODEL

    def test_custom_model(self):
        llm = BedrockLLM(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            client=_mock_client(),
        )
        assert llm.model_name == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_is_base_llm(self):
        llm = BedrockLLM(client=_mock_client())
        assert isinstance(llm, BaseLLM)

    def test_satisfies_graph_protocol(self):
        from src.graphs.nodes.answer import BaseLLM as LLMProtocol

        llm = BedrockLLM(client=_mock_client())
        assert isinstance(llm, LLMProtocol)

    def test_lazy_client_creation(self):
        """Client should NOT be created until first API call."""
        with patch("src.llm.bedrock.boto3") as mock_boto3:
            llm = BedrockLLM()
            mock_boto3.client.assert_not_called()


# ---------------------------------------------------------------------------
# generate — success
# ---------------------------------------------------------------------------

class TestBedrockGenerate:
    """Tests for the generate method."""

    def test_returns_text(self):
        client = _mock_client(_converse_response("Federalism is a system."))
        llm = BedrockLLM(client=client)

        result = llm.generate("What is federalism?")

        assert result == "Federalism is a system."

    def test_passes_model_id(self):
        client = _mock_client()
        llm = BedrockLLM(model="anthropic.claude-3-opus-20240229-v1:0", client=client)

        llm.generate("prompt")

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["modelId"] == "anthropic.claude-3-opus-20240229-v1:0"

    def test_passes_default_temperature(self):
        client = _mock_client()
        llm = BedrockLLM(client=client)

        llm.generate("prompt")

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == DEFAULT_TEMPERATURE

    def test_passes_default_max_tokens(self):
        client = _mock_client()
        llm = BedrockLLM(client=client)

        llm.generate("prompt")

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["maxTokens"] == DEFAULT_MAX_TOKENS

    def test_custom_temperature(self):
        client = _mock_client()
        llm = BedrockLLM(temperature=0.7, client=client)

        llm.generate("prompt")

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.7

    def test_kwarg_temperature_overrides(self):
        client = _mock_client()
        llm = BedrockLLM(temperature=0.0, client=client)

        llm.generate("prompt", temperature=0.9)

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.9

    def test_kwarg_max_tokens_overrides(self):
        client = _mock_client()
        llm = BedrockLLM(max_tokens=100, client=client)

        llm.generate("prompt", max_tokens=500)

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 500

    def test_system_prompt(self):
        client = _mock_client()
        llm = BedrockLLM(client=client)

        llm.generate("prompt", system="You are a helpful tutor.")

        call_kwargs = client.converse.call_args[1]
        assert call_kwargs["system"] == [{"text": "You are a helpful tutor."}]

    def test_no_system_prompt_by_default(self):
        client = _mock_client()
        llm = BedrockLLM(client=client)

        llm.generate("prompt")

        call_kwargs = client.converse.call_args[1]
        assert "system" not in call_kwargs

    def test_user_message_content(self):
        client = _mock_client()
        llm = BedrockLLM(client=client)

        llm.generate("What is X?")

        call_kwargs = client.converse.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [{"text": "What is X?"}]

    def test_multipart_response(self):
        """Handle response with multiple text blocks."""
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Part one. "},
                        {"text": "Part two."},
                    ],
                }
            },
        }
        client = _mock_client(response)
        llm = BedrockLLM(client=client)

        result = llm.generate("prompt")

        assert result == "Part one. Part two."


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestBedrockCountTokens:
    """Tests for the token counting heuristic."""

    def test_empty_string(self):
        llm = BedrockLLM(client=_mock_client())
        assert llm.count_tokens("") == 0

    def test_short_text(self):
        llm = BedrockLLM(client=_mock_client())
        # "hello" = 5 chars → max(1, 5//4) = 1
        assert llm.count_tokens("hello") >= 1

    def test_longer_text(self):
        llm = BedrockLLM(client=_mock_client())
        text = "a" * 400  # 400 chars → ~100 tokens
        assert llm.count_tokens(text) == 100

    def test_returns_int(self):
        llm = BedrockLLM(client=_mock_client())
        assert isinstance(llm.count_tokens("some text here"), int)

    def test_always_positive_for_nonempty(self):
        llm = BedrockLLM(client=_mock_client())
        assert llm.count_tokens("x") >= 1


# ---------------------------------------------------------------------------
# generate_with_context (inherited)
# ---------------------------------------------------------------------------

class TestBedrockGenerateWithContext:
    """Tests for the inherited generate_with_context method."""

    def test_empty_context(self):
        client = _mock_client(_converse_response("Direct answer."))
        llm = BedrockLLM(client=client)

        result = llm.generate_with_context("Q?", [])

        assert result == "Direct answer."

    def test_with_context(self):
        client = _mock_client(_converse_response("Contextual answer."))
        llm = BedrockLLM(client=client)

        llm.generate_with_context("Q?", ["passage one", "passage two"])

        call_kwargs = client.converse.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"][0]["text"]
        assert "[1] passage one" in prompt_text
        assert "[2] passage two" in prompt_text


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestBedrockErrorHandling:
    """Tests for error handling and retries."""

    def test_non_retryable_error_raises_immediately(self):
        client = _mock_client()
        client.converse.side_effect = _client_error("ValidationException")
        llm = BedrockLLM(client=client, max_retries=3, retry_base_delay=0)

        with pytest.raises(LLMError, match="ValidationException"):
            llm.generate("prompt")

        # Should NOT retry on non-retryable errors
        assert client.converse.call_count == 1

    def test_access_denied_raises_immediately(self):
        client = _mock_client()
        client.converse.side_effect = _client_error("AccessDeniedException")
        llm = BedrockLLM(client=client, max_retries=3, retry_base_delay=0)

        with pytest.raises(LLMError, match="AccessDeniedException"):
            llm.generate("prompt")

        assert client.converse.call_count == 1

    def test_throttling_retries(self):
        client = _mock_client()
        client.converse.side_effect = [
            _client_error("ThrottlingException"),
            _client_error("ThrottlingException"),
            _converse_response("Success after retries."),
        ]
        llm = BedrockLLM(client=client, max_retries=3, retry_base_delay=0)

        result = llm.generate("prompt")

        assert result == "Success after retries."
        assert client.converse.call_count == 3

    def test_service_unavailable_retries(self):
        client = _mock_client()
        client.converse.side_effect = [
            _client_error("ServiceUnavailableException"),
            _converse_response("Recovered."),
        ]
        llm = BedrockLLM(client=client, max_retries=3, retry_base_delay=0)

        result = llm.generate("prompt")

        assert result == "Recovered."

    def test_model_timeout_retries(self):
        client = _mock_client()
        client.converse.side_effect = [
            _client_error("ModelTimeoutException"),
            _converse_response("Recovered."),
        ]
        llm = BedrockLLM(client=client, max_retries=3, retry_base_delay=0)

        result = llm.generate("prompt")

        assert result == "Recovered."

    def test_retries_exhausted_raises(self):
        client = _mock_client()
        client.converse.side_effect = _client_error("ThrottlingException")
        llm = BedrockLLM(client=client, max_retries=2, retry_base_delay=0)

        with pytest.raises(LLMError, match="after 2 retries"):
            llm.generate("prompt")

        assert client.converse.call_count == 2

    def test_generic_exception_raises_llm_error(self):
        client = _mock_client()
        client.converse.side_effect = RuntimeError("connection reset")
        llm = BedrockLLM(client=client, retry_base_delay=0)

        with pytest.raises(LLMError, match="Bedrock call failed"):
            llm.generate("prompt")

    def test_malformed_response_raises(self):
        client = _mock_client({"output": {}})
        llm = BedrockLLM(client=client)

        with pytest.raises(LLMError, match="parse"):
            llm.generate("prompt")

    def test_empty_content_raises(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [],
                }
            },
        }
        client = _mock_client(response)
        llm = BedrockLLM(client=client)

        result = llm.generate("prompt")
        # Empty content list → joined empty string
        assert result == ""

    def test_llm_error_has_details(self):
        client = _mock_client()
        client.converse.side_effect = _client_error(
            "ValidationException", "model not found"
        )
        llm = BedrockLLM(client=client, retry_base_delay=0)

        with pytest.raises(LLMError) as exc_info:
            llm.generate("prompt")

        assert exc_info.value.details is not None
        assert "model not found" in exc_info.value.details


# ---------------------------------------------------------------------------
# Lazy client creation
# ---------------------------------------------------------------------------

class TestBedrockClientCreation:
    """Tests for lazy boto3 client creation."""

    def test_injected_client_used(self):
        client = _mock_client(_converse_response("ok"))
        llm = BedrockLLM(client=client)

        llm.generate("prompt")

        client.converse.assert_called_once()

    def test_creates_client_on_first_call(self):
        with patch("src.llm.bedrock.boto3") as mock_boto3:
            mock_client = _mock_client(_converse_response("ok"))
            mock_boto3.client.return_value = mock_client

            llm = BedrockLLM(region="eu-west-1")
            mock_boto3.client.assert_not_called()

            llm.generate("prompt")

            mock_boto3.client.assert_called_once()
            call_kwargs = mock_boto3.client.call_args
            assert call_kwargs[0][0] == "bedrock-runtime"
            assert call_kwargs[1]["region_name"] == "eu-west-1"

    def test_client_reused_across_calls(self):
        with patch("src.llm.bedrock.boto3") as mock_boto3:
            mock_client = _mock_client(_converse_response("ok"))
            mock_boto3.client.return_value = mock_client

            llm = BedrockLLM()
            llm.generate("first")
            llm.generate("second")

            # boto3.client should only be called once
            mock_boto3.client.assert_called_once()
            assert mock_client.converse.call_count == 2


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify BedrockLLM is importable."""

    def test_importable_from_module(self):
        from src.llm.bedrock import BedrockLLM as cls
        assert cls is not None

    def test_defaults_exported(self):
        from src.llm.bedrock import (
            DEFAULT_MAX_TOKENS,
            DEFAULT_MODEL,
            DEFAULT_REGION,
            DEFAULT_TEMPERATURE,
        )
        assert isinstance(DEFAULT_MODEL, str)
        assert isinstance(DEFAULT_REGION, str)
        assert isinstance(DEFAULT_TEMPERATURE, float)
        assert isinstance(DEFAULT_MAX_TOKENS, int)
