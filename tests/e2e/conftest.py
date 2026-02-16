"""Shared fixtures for end-to-end tests.

The ``--run-e2e`` flag and skip logic are registered in the root
``tests/conftest.py`` so they are available project-wide.
"""

import os
import tempfile
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture()
def sample_corpus(tmp_dir: str) -> str:
    """Create a small corpus of text files for pipeline testing.

    Returns the path to the corpus directory.
    """
    corpus_dir = os.path.join(tmp_dir, "corpus")
    os.makedirs(corpus_dir)

    files = {
        "federalism.txt": textwrap.dedent("""\
            Federalism is a system of government in which power is divided
            between a national (federal) government and various regional
            governments. In the United States, the federal government and
            the fifty state governments share authority. The Tenth Amendment
            to the Constitution reserves powers not granted to the federal
            government to the states or the people.
        """),
        "branches.txt": textwrap.dedent("""\
            The United States government is divided into three branches:
            the legislative branch (Congress), the executive branch (the
            President), and the judicial branch (the Supreme Court and
            lower federal courts). This separation of powers ensures that
            no single branch becomes too powerful. Congress makes laws,
            the President enforces laws, and the courts interpret laws.
        """),
        "bill_of_rights.txt": textwrap.dedent("""\
            The Bill of Rights comprises the first ten amendments to the
            United States Constitution. Ratified in 1791, these amendments
            guarantee essential rights and freedoms including freedom of
            speech, freedom of the press, the right to bear arms, and
            protection against unreasonable searches and seizures. The
            Bill of Rights was introduced by James Madison to address
            concerns about individual liberties.
        """),
    }

    for fname, content in files.items():
        with open(os.path.join(corpus_dir, fname), "w", encoding="utf-8") as f:
            f.write(content)

    return corpus_dir


class FakeLLM:
    """A deterministic fake LLM for e2e tests.

    Returns canned answers that echo back chunk content so that
    groundedness scoring works predictably.
    """

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a response based on keyword detection in the prompt."""
        prompt_lower = prompt.lower()

        if "federalism" in prompt_lower:
            return (
                "Federalism is a system of government in which power is "
                "divided between a national government and regional governments. "
                "The Tenth Amendment reserves powers to the states."
            )
        if "three branches" in prompt_lower or "branches of government" in prompt_lower:
            return (
                "The three branches of government are the legislative branch "
                "(Congress), the executive branch (the President), and the "
                "judicial branch (the Supreme Court). This separation of powers "
                "ensures no single branch becomes too powerful."
            )
        if "bill of rights" in prompt_lower:
            return (
                "The Bill of Rights comprises the first ten amendments to the "
                "Constitution. These amendments guarantee essential rights "
                "including freedom of speech and freedom of the press."
            )
        if "chocolate cake" in prompt_lower or "bitcoin" in prompt_lower:
            return ""

        # Default: echo a generic answer with some prompt keywords
        return f"Based on the provided context, the answer involves: {prompt[:100]}"

    def generate_with_context(
        self, prompt: str, context: list[str], **kwargs
    ) -> str:
        """Generate with context passages prepended."""
        combined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        full_prompt = f"Context:\n{combined}\n\nQuestion: {prompt}"
        return self.generate(full_prompt, **kwargs)

    def count_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return len(text.split())
