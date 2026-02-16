"""Route node for classifying incoming queries.

This node inspects the user's question and assigns a ``query_type``
that downstream nodes use to decide how (or whether) to answer.

Supported query types:
- **factual** — knowledge-seeking questions ("What is …", "Who …",
  "When …", "Define …", etc.).
- **procedural** — how-to / step-by-step questions ("How do I …",
  "How to …", "Steps to …", "Explain how …", etc.).
- **unsupported** — greetings, chitchat, commands, or anything that
  cannot be answered from a document corpus.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_node(state: GraphState) -> dict:
    """Classify the query in *state* and return the routing decision.

    The function examines ``state["question"]`` using lightweight regex
    heuristics and returns a dict with ``query_type`` set to one of
    ``"factual"``, ``"procedural"``, or ``"unsupported"``.

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
    # Check unsupported first — short-circuit greetings / chitchat.
    if _matches_any(query, _UNSUPPORTED_PATTERNS):
        return "unsupported"

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
