"""Tests for the BaseLLM abstract base class."""

from typing import Any

import pytest

from src.llm import BaseLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CompleteLLM(BaseLLM):
    """Concrete subclass implementing all abstract methods."""

    @property
    def model_name(self) -> str:
        return "complete-llm"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"response to: {prompt}"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class MissingGenerate(BaseLLM):
    """Subclass missing the generate method."""

    @property
    def model_name(self) -> str:
        return "incomplete"

    def count_tokens(self, text: str) -> int:
        return 0


class MissingCountTokens(BaseLLM):
    """Subclass missing the count_tokens method."""

    @property
    def model_name(self) -> str:
        return "incomplete"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return ""


class MissingModelName(BaseLLM):
    """Subclass missing the model_name property."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return ""

    def count_tokens(self, text: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# Cannot instantiate the ABC
# ---------------------------------------------------------------------------

class TestBaseLLMAbstract:
    """Tests that BaseLLM cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_error_mentions_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseLLM()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Subclass must implement all methods
# ---------------------------------------------------------------------------

class TestSubclassRequirements:
    """Tests that incomplete subclasses cannot be instantiated."""

    def test_missing_generate_raises(self):
        with pytest.raises(TypeError):
            MissingGenerate()  # type: ignore[abstract]

    def test_missing_count_tokens_raises(self):
        with pytest.raises(TypeError):
            MissingCountTokens()  # type: ignore[abstract]

    def test_missing_model_name_raises(self):
        with pytest.raises(TypeError):
            MissingModelName()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        llm = CompleteLLM()
        assert llm is not None


# ---------------------------------------------------------------------------
# Concrete subclass behaviour
# ---------------------------------------------------------------------------

class TestCompleteLLM:
    """Tests for a concrete BaseLLM subclass."""

    def test_model_name(self):
        llm = CompleteLLM()
        assert llm.model_name == "complete-llm"

    def test_generate(self):
        llm = CompleteLLM()
        result = llm.generate("hello")
        assert "hello" in result

    def test_generate_with_kwargs(self):
        llm = CompleteLLM()
        result = llm.generate("hello", temperature=0.5, max_tokens=100)
        assert isinstance(result, str)

    def test_count_tokens(self):
        llm = CompleteLLM()
        count = llm.count_tokens("one two three")
        assert count == 3

    def test_count_tokens_empty(self):
        llm = CompleteLLM()
        assert llm.count_tokens("") == 0

    def test_count_tokens_returns_int(self):
        llm = CompleteLLM()
        assert isinstance(llm.count_tokens("text"), int)


# ---------------------------------------------------------------------------
# generate_with_context default implementation
# ---------------------------------------------------------------------------

class TestGenerateWithContext:
    """Tests for the default generate_with_context method."""

    def test_empty_context_delegates_to_generate(self):
        llm = CompleteLLM()
        result = llm.generate_with_context("What is X?", [])
        # Should just call generate directly
        assert "What is X?" in result

    def test_single_context(self):
        llm = CompleteLLM()
        result = llm.generate_with_context("Q?", ["passage one"])
        assert "[1] passage one" in result

    def test_multiple_contexts_numbered(self):
        llm = CompleteLLM()
        result = llm.generate_with_context("Q?", ["alpha", "beta", "gamma"])
        assert "[1] alpha" in result
        assert "[2] beta" in result
        assert "[3] gamma" in result

    def test_includes_question(self):
        llm = CompleteLLM()
        result = llm.generate_with_context("What is federalism?", ["ctx"])
        assert "What is federalism?" in result

    def test_kwargs_forwarded(self):
        """Ensure kwargs are passed through to generate."""

        class RecordingLLM(BaseLLM):
            def __init__(self):
                self.last_kwargs: dict = {}

            @property
            def model_name(self) -> str:
                return "recording"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                self.last_kwargs = kwargs
                return "ok"

            def count_tokens(self, text: str) -> int:
                return 0

        llm = RecordingLLM()
        llm.generate_with_context("Q?", ["ctx"], temperature=0.7, stop=["\n"])
        assert llm.last_kwargs["temperature"] == 0.7
        assert llm.last_kwargs["stop"] == ["\n"]

    def test_can_be_overridden(self):
        """Subclasses can override generate_with_context."""

        class CustomContextLLM(BaseLLM):
            @property
            def model_name(self) -> str:
                return "custom"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                return prompt

            def count_tokens(self, text: str) -> int:
                return 0

            def generate_with_context(
                self, prompt: str, context: list[str], **kwargs: Any
            ) -> str:
                return f"custom:{prompt}"

        llm = CustomContextLLM()
        result = llm.generate_with_context("Q?", ["ctx"])
        assert result == "custom:Q?"


# ---------------------------------------------------------------------------
# Protocol compatibility
# ---------------------------------------------------------------------------

class TestProtocolCompatibility:
    """Verify that BaseLLM subclasses satisfy the graph node protocol."""

    def test_satisfies_answer_node_protocol(self):
        from src.graphs.nodes.answer import BaseLLM as LLMProtocol

        llm = CompleteLLM()
        assert isinstance(llm, LLMProtocol)

    def test_has_generate_method(self):
        llm = CompleteLLM()
        assert callable(getattr(llm, "generate", None))

    def test_has_model_name_property(self):
        llm = CompleteLLM()
        assert isinstance(llm.model_name, str)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify BaseLLM is importable from the llm package."""

    def test_base_llm_exported(self):
        from src.llm import BaseLLM as cls
        assert cls is not None

    def test_is_abstract(self):
        import inspect
        assert inspect.isabstract(BaseLLM)
