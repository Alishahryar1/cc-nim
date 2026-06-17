"""Keyword-relevance tool culling.

Free models drown in 91-tool inventories. opencode loads ~41 and works fine.
This module slims a request's tool list down to the ones most relevant to the
user's latest question, by scoring each tool's name+description against tokens
extracted from recent messages.

Strategy: simple TF-style overlap. No embedding model, no extra calls. Tools
that are universally useful (Read, Write, Edit, Bash, etc.) are pinned via
``ALWAYS_KEEP`` regardless of score so the agent never loses fundamentals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


# Names of tools the agent always needs, regardless of relevance score.
# Matched case-insensitively, prefix match.
ALWAYS_KEEP = (
    "read",
    "write",
    "edit",
    "bash",
    "grep",
    "glob",
    "ls",
    "todowrite",
    "taskcreate",
    "exitplanmode",
    "askuserquestion",
    "webfetch",
    "websearch",
)


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "you", "your", "from",
        "have", "has", "are", "was", "will", "just", "can", "would", "could",
        "should", "what", "how", "why", "when", "where", "which", "into",
        "out", "use", "using", "used", "make", "made", "get", "got", "set",
        "let", "see", "look", "show", "tell", "say", "does", "did", "doing",
        "but", "not", "all", "any", "some", "more", "most", "less", "now",
        "also", "very", "really", "still", "yet", "only", "even", "than",
    }
)


@dataclass(slots=True)
class CullResult:
    """Outcome of a culling pass."""

    kept_indices: tuple[int, ...]
    dropped_count: int


def _tokenize(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _TOKEN_RE.finditer(text)
        if match.group(0).lower() not in _STOPWORDS
    }


def _tool_text(tool: Any) -> str:
    name = getattr(tool, "name", "") or ""
    description = getattr(tool, "description", "") or ""
    return f"{name} {description}"


def _is_always_keep(tool: Any) -> bool:
    name = (getattr(tool, "name", "") or "").lower()
    return any(name == keep or name.startswith(keep + "_") for keep in ALWAYS_KEEP)


def _extract_intent_text(messages: list[Any], system: Any) -> str:
    """Combine the last user-text + system prompt into one intent string."""
    chunks: list[str] = []
    if isinstance(system, str):
        chunks.append(system)
    elif isinstance(system, list):
        for item in system:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    chunks.append(t)
    # Last few user messages, prefer text blocks over tool_results.
    user_texts: list[str] = []
    for message in reversed(messages):
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            user_texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "text":
                    text = getattr(block, "text", "")
                    if text:
                        user_texts.append(text)
        if len(user_texts) >= 3:
            break
    chunks.extend(reversed(user_texts))
    return " ".join(chunks)


def cull_tools(
    tools: list[Any],
    messages: list[Any],
    system: Any,
    *,
    max_keep: int,
) -> CullResult:
    """Return indices into ``tools`` to keep; drop the rest.

    Indices are sorted ascending so the caller can rebuild a stable tools list.
    If ``len(tools) <= max_keep``, returns all indices unchanged.
    """
    if len(tools) <= max_keep:
        return CullResult(
            kept_indices=tuple(range(len(tools))), dropped_count=0
        )

    intent_tokens = _tokenize(_extract_intent_text(messages, system))
    pinned: list[int] = []
    scored: list[tuple[int, int]] = []
    for idx, tool in enumerate(tools):
        if _is_always_keep(tool):
            pinned.append(idx)
            continue
        tool_tokens = _tokenize(_tool_text(tool))
        score = len(intent_tokens & tool_tokens) if intent_tokens else 0
        scored.append((idx, score))

    # Sort scored by score desc, original index asc as tiebreaker for stability.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    remaining_slots = max(0, max_keep - len(pinned))
    chosen = sorted(pinned + [idx for idx, _ in scored[:remaining_slots]])
    return CullResult(
        kept_indices=tuple(chosen), dropped_count=len(tools) - len(chosen)
    )


def apply_cull(tools: list[Any], result: CullResult) -> list[Any]:
    """Rebuild the tools list using only kept indices."""
    if result.dropped_count == 0:
        return tools
    return [tools[i] for i in result.kept_indices]
