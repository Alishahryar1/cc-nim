"""Provider-specific exception mapping."""

import httpx
import openai

from core.anthropic import get_user_facing_error_message
from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)
from providers.rate_limit import GlobalRateLimiter


def user_visible_message_for_mapped_provider_error(
    mapped: Exception,
    *,
    provider_name: str,
    read_timeout_s: float | None,
) -> str:
    """Return the user-visible string after :func:`map_error` (405 + mapped types)."""
    if getattr(mapped, "status_code", None) == 405:
        return (
            f"Upstream provider {provider_name} rejected the request method "
            "or endpoint (HTTP 405)."
        )
    return get_user_facing_error_message(mapped, read_timeout_s=read_timeout_s)


def map_error(
    e: Exception, *, rate_limiter: GlobalRateLimiter | None = None
) -> Exception:
    """Map OpenAI or HTTPX exception to specific ProviderError.

    Streaming transports should pass their scoped limiter (``self._global_rate_limiter``)
    so reactive 429 handling applies to the correct provider. Tests may omit
    ``rate_limiter`` to use the process-wide singleton.
    """
    message = get_user_facing_error_message(e)
    limiter = rate_limiter or GlobalRateLimiter.get_instance()

    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(message, raw_error=str(e))
    if isinstance(e, openai.RateLimitError):
        limiter.set_blocked(60)
        return RateLimitError(message, raw_error=str(e))
    if isinstance(e, openai.BadRequestError):
        return InvalidRequestError(message, raw_error=str(e))
    if isinstance(e, openai.InternalServerError):
        raw_message = str(e)
        sdk_status = getattr(e, "status_code", None)
        if "overloaded" in raw_message.lower() or "capacity" in raw_message.lower():
            return OverloadedError(message, raw_error=raw_message)
        if isinstance(sdk_status, int) and 500 <= sdk_status <= 599:
            stable = APIError("_", status_code=sdk_status)
            return APIError(
                get_user_facing_error_message(stable),
                status_code=sdk_status,
                raw_error=str(e),
            )
        return APIError(message, status_code=500, raw_error=str(e))
    if isinstance(
        e,
        (
            openai.APITimeoutError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
        ),
    ):
        # Return as-is; user_visible_message_for_mapped_provider_error will format correctly
        return e

    if isinstance(e, openai.APIError):
        return APIError(
            message, status_code=getattr(e, "status_code", 500), raw_error=str(e)
        )

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status in (401, 403):
            return AuthenticationError(message, raw_error=str(e))
        if status == 429:
            limiter.set_blocked(60)
            return RateLimitError(message, raw_error=str(e))
        if status == 400:
            return InvalidRequestError(message, raw_error=str(e))
        if status >= 500:
            if status in (502, 503, 504):
                return OverloadedError(message, raw_error=str(e))
            return APIError(message, status_code=status, raw_error=str(e))
        return APIError(message, status_code=status, raw_error=str(e))

    return e
