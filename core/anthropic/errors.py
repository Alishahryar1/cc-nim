"""User-facing error formatting shared by API, providers, and integrations."""

import json

import httpx
import openai


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
    if isinstance(e, (httpx.RemoteProtocolError, httpx.ReadError)):
        return "Provider connection dropped unexpectedly."
    if isinstance(e, httpx.ConnectTimeout):
        return "Could not connect to provider."
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
    if isinstance(e, openai.APIConnectionError):
        return "Provider connection dropped unexpectedly."

    if isinstance(e, json.JSONDecodeError):
        return (
            "Provider returned an empty or invalid response. "
            "This may indicate the model does not support the request format "
            "(e.g. image inputs sent to a text-only model)."
        )

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
