"""Recover from upstream ``max_(completion_)tokens`` too-large 400 rejections.

Some OpenAI-compatible providers (Groq, NVIDIA NIM, ...) cap the per-request
output token count below what Claude Code asks for and reject the whole request
with an HTTP 400 that names the allowed maximum, e.g.::

    max_completion_tokens must be less than or equal to 40960, ...

This module parses that maximum and clamps the request body so the transport can
retry once and succeed. The transport also remembers the learned cap per model
so later requests clamp proactively instead of paying the 400 every time.
"""

import json
import re
from typing import Any

import openai

# Body keys that carry the output-token budget across OpenAI-compatible policies.
_OUTPUT_TOKEN_FIELDS = ("max_completion_tokens", "max_tokens")

# Only treat a 400 as an output-cap rejection when it names one of these fields.
_OUTPUT_TOKEN_KEYWORDS = ("max_completion_tokens", "max_tokens")

# Comparator phrases that precede the allowed maximum in provider error text.
_CAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"less than or equal to\s+(\d+)"),
    re.compile(r"smaller than or equal to\s+(\d+)"),
    re.compile(r"<=\s*(\d+)"),
    re.compile(r"at most\s+(\d+)"),
    re.compile(r"must not exceed\s+(\d+)"),
    re.compile(r"maximum(?:\s+value)?(?:\s+for\s+\S+)?\s+is\s+(\d+)"),
    re.compile(r"maximum(?:\s+allowed)?(?:\s+value)?\s+of\s+(\d+)"),
)


def _is_bad_request(error: Exception) -> bool:
    return isinstance(error, openai.BadRequestError) or (
        getattr(error, "status_code", None) == 400
    )


def _error_text(error: Exception) -> str:
    text = str(error)
    body = getattr(error, "body", None)
    if body is not None:
        text = f"{text} {json.dumps(body, default=str)}"
    return text.lower()


def parse_output_token_cap(error: Exception) -> int | None:
    """Return the allowed output-token maximum named in a 400 rejection, if any."""
    if not _is_bad_request(error):
        return None

    text = _error_text(error)
    if not any(keyword in text for keyword in _OUTPUT_TOKEN_KEYWORDS):
        return None

    for pattern in _CAP_PATTERNS:
        match = pattern.search(text)
        if match:
            cap = int(match.group(1))
            if cap > 0:
                return cap
    return None


def clamp_output_tokens(body: dict[str, Any], cap: int) -> dict[str, Any] | None:
    """Return a shallow clone with output-token fields clamped to ``cap``.

    Returns ``None`` when nothing needs clamping (no output field exceeds the
    cap), so callers can avoid a pointless identical retry.
    """
    clamped: dict[str, Any] | None = None
    for field in _OUTPUT_TOKEN_FIELDS:
        value = body.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > cap:
            if clamped is None:
                clamped = dict(body)
            clamped[field] = cap
    return clamped
