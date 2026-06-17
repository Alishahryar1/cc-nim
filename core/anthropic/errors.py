"""User-facing error formatting shared by API, providers, and integrations."""

import re

import httpx
import openai


def _redact_token(text: str) -> str:
    # Redact header-style values: Authorization / api_key / token / secret / password
    text = re.sub(
        r"(?i)((Authorization|api_key|X-Api-Key|X-API-Key|token|secret|password)[^\n]*?)(\s*[=:]\s*)([^\n]+)",
        lambda m: f"{m.group(1)}{m.group(3)}[REDACTED]",
        text,
    )
    # Redact long opaque tokens.
    text = re.sub(r"\b[A-Za-z0-9_+/=-]{32,}\b", "[REDACTED_TOKEN]", text)
    return text


def _safe_raw_error_preview(raw_error: object, max_len: int = 180) -> str:
    if raw_error is None:
        return ""
    text = str(raw_error).strip()
    if not text:
        return ""
    first_line = text.splitlines()[0]
    first_line = _redact_token(first_line)
    return first_line[:max_len]


def _format_auth_error(base_message: str, exc: object) -> str:
    if not isinstance(exc, Exception):
        return base_message
    preview = _safe_raw_error_preview(getattr(exc, "raw_error", None))
    if not preview:
        return base_message
    return f"{base_message} Upstream recap: {preview}"


def get_user_facing_error_message(
    e: Exception,
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
    if isinstance(e, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."

    if isinstance(e, openai.RateLimitError):
        return "Provider rate limit reached. Please retry shortly."
    if isinstance(e, openai.AuthenticationError):
        return _format_auth_error("Provider authentication failed. Check API key.", e)
    if isinstance(e, openai.BadRequestError):
        return "Invalid request sent to provider."

    name = type(e).__name__
    status_code = getattr(e, "status_code", None)
    if name == "RateLimitError":
        return "Provider rate limit reached. Please retry shortly."
    if name == "AuthenticationError":
        return _format_auth_error("Provider authentication failed. Check API key.", e)
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

    message = str(e).strip()
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
