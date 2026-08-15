"""Recover from upstream ``max_(completion_)tokens`` too-large 400 rejections.

Some OpenAI-compatible providers (Groq, NVIDIA NIM, ...) cap the per-request
output token count below what Claude Code asks for and reject the whole request
with an HTTP 400 that names the allowed maximum, e.g.::

    max_completion_tokens must be less than or equal to 40960, ...

This module parses that maximum and clamps the request body so the provider can
retry once and succeed. The provider also remembers the learned cap per model
so later requests clamp proactively instead of paying the 400 every time.
"""

import json
import re
from typing import Any

import openai

# Body keys that carry the output-token budget across OpenAI-compatible policies.
_OUTPUT_TOKEN_FIELDS = ("max_completion_tokens", "max_tokens")

_CAP_VALUE_PATTERN = r"[`'\"]?(\d+)[`'\"]?"
_OUTPUT_TOKEN_FIELD_PATTERN = r"[`'\"]?(?:max_completion_tokens|max_tokens)[`'\"]?"

# An accepted grammar must bind the output-token field and its numeric limit.
# A field mentioned elsewhere in the response must never authorize an unrelated
# comparator.
_FIELD_BOUND_CAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"range\s+of\s+{_OUTPUT_TOKEN_FIELD_PATTERN}\s+should\s+be\s+"
        rf"\[\s*1\s*,\s*{_CAP_VALUE_PATTERN}\s*\]"
    ),
    re.compile(
        rf"{_OUTPUT_TOKEN_FIELD_PATTERN}\s*(?::\s*)?"
        rf"(?:"
        rf"(?:(?:must|should)\s+be\s+)?"
        rf"(?:less|smaller)\s+than\s+or\s+equal\s+to"
        rf"|(?:(?:must|should)\s+be\s*)?<="
        rf"|(?:(?:must|should)\s+be\s+)?at\s+most"
        rf"|(?:must|should)\s+not\s+exceed"
        rf"|maximum(?:\s+allowed)?(?:\s+value)?\s+(?:is|of)"
        rf")\s*{_CAP_VALUE_PATTERN}"
    ),
    re.compile(
        rf"maximum(?:\s+allowed)?(?:\s+value)?\s+for\s+"
        rf"{_OUTPUT_TOKEN_FIELD_PATTERN}\s+is\s+{_CAP_VALUE_PATTERN}"
    ),
    re.compile(
        rf"maximum(?:\s+allowed)?(?:\s+value)?\s+of\s+"
        rf"{_CAP_VALUE_PATTERN}\s+for\s+{_OUTPUT_TOKEN_FIELD_PATTERN}"
    ),
)


def _is_bad_request(error: Exception) -> bool:
    return isinstance(error, openai.BadRequestError) or (
        getattr(error, "status_code", None) == 400
    )


def _error_texts(error: Exception) -> tuple[str, ...]:
    """Normalize free-form and structured errors into field-bound text."""
    texts = [str(error)]
    body = getattr(error, "body", None)
    if body is not None:
        texts.append(json.dumps(body, default=str))

    pending: list[Any] = [body]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            param = value.get("param")
            message = value.get("message")
            if (
                isinstance(param, str)
                and param.strip("`'\" ").lower() in _OUTPUT_TOKEN_FIELDS
                and isinstance(message, str)
            ):
                # Prefixing an explicitly named structured parameter lets the
                # same field-bound grammar handle comparator-only messages.
                texts.append(f"{param} {message}")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return tuple(text.lower() for text in texts)


def _parse_cap(text: str) -> int | None:
    for pattern in _FIELD_BOUND_CAP_PATTERNS:
        match = pattern.search(text)
        if match:
            cap = int(match.group(1))
            if cap > 0:
                return cap
    return None


def parse_output_token_cap(error: Exception) -> int | None:
    """Return the allowed output-token maximum named in a 400 rejection, if any."""
    if not _is_bad_request(error):
        return None

    for text in _error_texts(error):
        cap = _parse_cap(text)
        if cap is not None:
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
