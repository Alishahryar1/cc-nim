"""Anthropic error shaping shared by API, providers, and integrations."""

import re
from typing import Any

import httpx
import openai

_SECRET_TEXT_REPLACEMENTS = (
    (
        re.compile(
            r"(?i)(?P<prefix>[\"']?authorization[\"']?\s*[:=]\s*)"
            r"(?P<quote>[\"']?)(?:(?:bearer|basic)\s+)?"
            r"[^\"'\s,;&}\]]+(?P=quote)"
        ),
        r"\g<prefix>\g<quote><redacted>\g<quote>",
    ),
    (
        re.compile(
            r"(?i)(?P<prefix>[\"']?(?:api[_-]?key|access[_-]?token|"
            r"refresh[_-]?token|token|client[_-]?secret|secret|password)"
            r"[\"']?\s*[:=]\s*)(?P<quote>[\"']?)"
            r"[^\"'\s,;&}\]]+(?P=quote)"
        ),
        r"\g<prefix>\g<quote><redacted>\g<quote>",
    ),
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1<redacted>"),
    (
        re.compile(
            r"(?i)(?<![a-z0-9])(?:sk-[a-z0-9._-]{8,}|"
            r"nvapi-[a-z0-9._-]{8,}|hf_[a-z0-9_-]{8,}|"
            r"gsk_[a-z0-9_-]{8,}|github_pat_[a-z0-9_]{8,}|"
            r"gh[pousr]_[a-z0-9]{8,}|AIza[a-z0-9_-]{20,})"
            r"(?![a-z0-9])"
        ),
        "<redacted>",
    ),
)

_ANTHROPIC_ERROR_STATUS_CODES = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def redact_sensitive_error_text(text: str) -> str:
    """Redact recognizable credentials while preserving diagnostic context."""
    sanitized = text
    for pattern, replacement in _SECRET_TEXT_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def anthropic_error_payload(
    *, error_type: str, message: str, request_id: str | None = None
) -> dict[str, Any]:
    """Return one Anthropic-compatible JSON error envelope."""
    payload: dict[str, Any] = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": redact_sensitive_error_text(message),
        },
    }
    if request_id:
        payload["request_id"] = request_id
    return payload


def anthropic_status_for_error_type(error_type: str) -> int:
    """Return the standard HTTP status for an Anthropic error type."""
    return _ANTHROPIC_ERROR_STATUS_CODES.get(error_type, 500)


def get_user_facing_error_message(
    e: BaseException,
    *,
    read_timeout_s: float | None = None,
) -> str:
    """Return a readable, non-empty error message for users.

    Known transport and OpenAI SDK exception types are mapped to stable wording
    before falling back to ``str(e)``, so empty or noisy SDK messages do not skip
    the mapped path.
    """
    if isinstance(e, httpx.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Provider request timed out."
    if isinstance(e, httpx.ConnectTimeout):
        return "Could not connect to provider."
    if isinstance(e, httpx.ConnectError):
        return "Could not connect to provider."
    if isinstance(e, httpx.RemoteProtocolError):
        return "Provider connection was interrupted before a response was received."
    if isinstance(e, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."

    if isinstance(e, openai.RateLimitError):
        return "Provider rate limit reached. Please retry shortly."
    if isinstance(e, openai.AuthenticationError):
        return "Provider authentication failed. Check API key."
    if isinstance(e, openai.BadRequestError):
        return "Invalid request sent to provider."

    name = type(e).__name__
    status_code = getattr(e, "status_code", None)
    if name == "RateLimitError":
        return "Provider rate limit reached. Please retry shortly."
    if name == "AuthenticationError":
        return "Provider authentication failed. Check API key."
    if name == "InvalidRequestError":
        return "Invalid request sent to provider."
    if name == "OverloadedError":
        return "Provider is currently overloaded. Please retry."
    if name == "APIError":
        if status_code in (502, 503, 504):
            return "Provider is temporarily unavailable. Please retry."
        return "Provider API request failed."
    if name.endswith("ProviderError") or name == "ProviderError":
        return "Provider request failed."

    message = redact_sensitive_error_text(str(e).strip())
    if message:
        return message

    return "Provider request failed unexpectedly."


def format_user_error_preview(exc: Exception, *, max_len: int = 200) -> str:
    """Truncate a user-facing error string for short chat replies."""
    return get_user_facing_error_message(exc)[:max_len]


def append_request_id(message: str, request_id: str | None) -> str:
    """Append request_id suffix when available."""
    base = message.strip() or "Provider request failed unexpectedly."
    if request_id:
        return f"{base} (request_id={request_id})"
    return base
