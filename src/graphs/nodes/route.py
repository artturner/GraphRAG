"""Route node for classifying incoming queries.

This node inspects the user's question and assigns a ``query_type``
that downstream nodes use to decide how (or whether) to answer.

Supported query types:
- **factual** — knowledge-seeking questions ("What is …", "Who …",
  "When …", "Define …", etc.).
- **procedural** — how-to / step-by-step questions ("How do I …",
  "How to …", "Steps to …", "Explain how …", etc.).
- **synthesis** — generative / creative requests ("Suggest …",
  "Brainstorm …", "Propose …", "List some ideas …", etc.).
- **summarize** — condensing requests ("Summarize …", "Key points of …",
  "TL;DR …", "Overview of …", etc.).
- **unsupported** — greetings, chitchat, commands, structural references
  (chapter N, page N, section N), or anything that cannot be answered
  from a document corpus via semantic search.
"""

import logging
import re

from src.graphs.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Patterns that indicate a factual, knowledge-seeking query.
_FACTUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^what\b",
        r"^who\b",
        r"^when\b",
        r"^where\b",
        r"^which\b",
        r"^why\b",
        r"^is\b",
        r"^are\b",
        r"^was\b",
        r"^were\b",
        r"^does\b",
        r"^do\b",
        r"^did\b",
        r"^can\b",
        r"^could\b",
        r"^has\b",
        r"^have\b",
        r"^had\b",
        r"^will\b",
        r"^would\b",
        r"^should\b",
        r"\bdefine\b",
        r"\bdefinition\b",
        r"\bdescribe\b",
        r"\bmeaning\b",
        r"\bwhat\s+(?:is|are|was|were|does|do)\b",
        r"\btell\s+me\s+about\b",
        r"\bexplain\s+(?:the|what|why)\b",
    ]
]

# Patterns that indicate a procedural / how-to query.
_PROCEDURAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^how\s+(?:do|can|to|should|would|could)\b",
        r"^how\s+is\b",
        r"^how\s+are\b",
        r"\bsteps?\s+to\b",
        r"\bprocess\s+(?:of|for|to)\b",
        r"\bprocedure\s+(?:of|for|to)\b",
        r"\bhow\s+(?:do\s+(?:i|you|we)\b)",
        r"\bguide\s+(?:to|for|on)\b",
        r"\binstructions?\s+(?:for|to|on)\b",
        r"\btutorial\b",
        r"\bwalkthrough\b",
        r"\bexplain\s+how\b",
    ]
]

# Patterns that indicate a summarization query.
_SUMMARIZE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^summarize\b",
        r"^summarise\b",
        r"^recap\b",
        r"^condense\b",
        r"\bsummar(?:y|ize|ise)\s+(?:of|the|this|chapter)\b",
        r"\bkey\s+(?:points?|ideas?|takeaways?|concepts?)\b",
        r"\bmain\s+(?:points?|ideas?|concepts?|themes?)\b",
        r"\boverview\s+of\b",
        r"\btl;?dr\b",
        r"\bin\s+(?:brief|short|summary|a\s+nutshell)\b",
        r"\bbrief(?:ly)?\s+(?:describe|explain|outline|summarize)\b",
        r"\bkey\s+takeaways?\b",
    ]
]

# Patterns that indicate a synthesis / generative query.
_SYNTHESIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^suggest\b",
        r"^brainstorm\b",
        r"^generate\b",
        r"^propose\b",
        r"^come\s+up\s+with\b",
        r"^recommend\b",
        r"^create\s+a?\s*list\b",
        r"^draft\b",
        r"^design\b",
        r"\blist\s+(?:some|possible|potential|research|ideas?)\b",
        r"\bsuggest\s+(?:some|a\s+few|possible|potential|research)\b",
        r"\bbrainstorm\b",
        r"\bideas?\s+for\b",
        r"\btopics?\s+(?:for|to|about)\b",
        r"\bpossible\s+(?:topics?|ideas?|directions?|approaches?)\b",
    ]
]

# Patterns that strongly signal an unsupported / off-topic query.
_UNSUPPORTED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b",
        r"^(?:thanks|thank\s+you|bye|goodbye|see\s+you)\b",
        r"^(?:help|stop|quit|exit|cancel)\s*$",
        r"^(?:ok|okay|sure|yes|no|maybe)\s*$",
        r"^(?:lol|haha|hehe|hmm+|wow|oh)\s*$",
    ]
]

# Patterns that reference document structure (chapter/page numbers) which
# cannot be resolved by semantic search alone — routed to refuse.
_STRUCTURAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bchapter\s+\d+\b",
        r"\bch\.?\s*\d+\b",
        r"\bpage\s+\d+\b",
        r"\bp\.\s*\d+\b",
        r"\bsection\s+\d+[\.\d]*\b",
    ]
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_node(state: GraphState) -> dict:
    """Classify the query in *state* and return the routing decision.

    The function examines ``state["question"]`` using lightweight regex
    heuristics and returns a dict with ``query_type`` set to one of
    ``"factual"``, ``"procedural"``, ``"synthesis"``, or ``"unsupported"``.

    Args:
        state: Current graph state — must contain ``question``.

    Returns:
        A dict with a single key ``query_type``.

    Example:
        ```python
        result = route_node({"question": "What is federalism?"})
        assert result == {"query_type": "factual"}
        ```
    """
    question: str = state.get("question", "")
    query = question.strip()

    if not query:
        logger.warning("route_node received empty question")
        return {"query_type": "unsupported"}

    query_type = _classify(query)

    logger.debug(
        "Classified query as %s: %s",
        query_type,
        query[:80],
    )

    return {"query_type": query_type}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify(query: str) -> str:
    """Return the query type for *query*."""
    # Structural references (chapter N, page N) can't be resolved by semantic
    # search — refuse early before retrieval wastes time or fabricates.
    if _matches_any(query, _STRUCTURAL_PATTERNS):
        return "unsupported"

    # Check unsupported next — short-circuit greetings / chitchat.
    if _matches_any(query, _UNSUPPORTED_PATTERNS):
        return "unsupported"

    # Summarize before synthesis — more specific intent.
    if _matches_any(query, _SUMMARIZE_PATTERNS):
        return "summarize"

    # Synthesis before procedural/factual — generative requests are distinct.
    if _matches_any(query, _SYNTHESIS_PATTERNS):
        return "synthesis"

    # Procedural before factual — "how" questions are more specific.
    if _matches_any(query, _PROCEDURAL_PATTERNS):
        return "procedural"

    if _matches_any(query, _FACTUAL_PATTERNS):
        return "factual"

    # If the query ends with '?' it's likely a question we can attempt.
    if query.rstrip().endswith("?"):
        return "factual"

    # Default: treat as unsupported so downstream nodes can refuse
    # gracefully rather than hallucinate.
    return "unsupported"


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return ``True`` if *text* matches any pattern in *patterns*."""
    return any(p.search(text) for p in patterns)
