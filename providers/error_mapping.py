"""Provider-specific exception mapping and user-visible diagnostics."""

import json
from dataclasses import dataclass
from typing import Any

import httpx
import openai

from config.constants import PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES
from core.anthropic import get_user_facing_error_message
from core.anthropic.streaming import (
    is_transient_overload_error,
    retryable_transient_status,
)
from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)
from providers.rate_limit import GlobalRateLimiter

_BODY_ATTR = "_fcc_provider_error_body"
_BODY_TRUNCATED_ATTR = "_fcc_provider_error_body_truncated"


@dataclass(frozen=True)
class ProviderErrorDetail:
    """Structured upstream error detail surfaced directly to users."""

    status_code: int | None = None
    body_text: str | None = None
    exception_text: str | None = None
    error_type_hint: str | None = None
    body_truncated: bool = False


def attach_provider_error_body(
    exc: Exception, body: bytes | str, *, truncated: bool = False
) -> None:
    """Attach a streamed HTTP error body to an exception for later formatting."""
    setattr(exc, _BODY_ATTR, body)
    setattr(exc, _BODY_TRUNCATED_ATTR, truncated)


def _status_code_from_exception(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _body_from_response(exc: Exception) -> Any:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return response.json()
    except ValueError, RuntimeError:
        pass
    try:
        return response.text
    except httpx.ResponseNotRead, RuntimeError:
        return None


def _normalize_body_text(body: Any) -> str | None:
    if body is None:
        return None
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    elif isinstance(body, str):
        text = body
    else:
        try:
            return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            text = str(body)
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return stripped
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _cap_text_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    capped = encoded[:max_bytes].decode("utf-8", errors="replace")
    return f"{capped}\n... [truncated after {max_bytes} bytes]", True


def _error_type_hint_from_body(body: Any, body_text: str | None) -> str | None:
    parsed = body
    if isinstance(parsed, bytes):
        text = parsed.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
    elif isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            for key in ("type", "code"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("type", "code"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if (
        body_text
        and "model" in body_text.lower()
        and "unsupported" in body_text.lower()
    ):
        return "upstream_model_error"
    return None


def extract_provider_error_detail(exc: Exception) -> ProviderErrorDetail:
    """Extract copyable upstream status/body/exception detail from provider errors."""
    raw_body = getattr(exc, _BODY_ATTR, None)
    raw_body_truncated = bool(getattr(exc, _BODY_TRUNCATED_ATTR, False))
    if raw_body is None:
        raw_body = getattr(exc, "body", None)
    if raw_body is None:
        raw_body = _body_from_response(exc)

    body_text = _normalize_body_text(raw_body)
    display_truncated = raw_body_truncated
    if body_text is not None:
        body_text, cap_truncated = _cap_text_bytes(
            body_text, PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES
        )
        display_truncated = display_truncated or cap_truncated

    exception_text = str(exc).strip() or None
    if exception_text is not None:
        exception_text, _ = _cap_text_bytes(
            exception_text, PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES
        )

    return ProviderErrorDetail(
        status_code=_status_code_from_exception(exc),
        body_text=body_text,
        exception_text=exception_text,
        error_type_hint=_error_type_hint_from_body(raw_body, body_text),
        body_truncated=display_truncated,
    )


def _provider_error_category(mapped: Exception) -> str | None:
    error_type = getattr(mapped, "error_type", None)
    if isinstance(error_type, str) and error_type.strip():
        return error_type.strip()
    return None


def _append_request_id_lines(lines: list[str], request_id: str | None) -> None:
    if request_id:
        lines.extend(("", f"Request ID: {request_id}"))


def format_provider_error_message(
    mapped: Exception,
    detail: ProviderErrorDetail,
    *,
    provider_name: str,
    read_timeout_s: float | None,
    request_id: str | None = None,
) -> str:
    """Return a copyable user-facing provider error including upstream detail."""
    stable_message = get_user_facing_error_message(
        mapped, read_timeout_s=read_timeout_s
    )
    has_upstream_detail = detail.status_code is not None or detail.body_text is not None
    if not has_upstream_detail:
        lines = [stable_message]
        if detail.exception_text and detail.exception_text != stable_message:
            lines.extend(("", "Provider exception:", detail.exception_text))
        _append_request_id_lines(lines, request_id)
        return "\n".join(lines)

    if detail.status_code == 405:
        lines = [
            f"Upstream provider {provider_name} rejected the request method "
            "or endpoint (HTTP 405)."
        ]
    elif detail.status_code is not None:
        lines = [
            f"Upstream provider {provider_name} returned HTTP {detail.status_code}."
        ]
    else:
        lines = [f"Upstream provider {provider_name} returned an error."]

    category = detail.error_type_hint or _provider_error_category(mapped)
    if category:
        lines.append(f"Category: {category}")
    if stable_message and stable_message != lines[0]:
        lines.append(f"Mapped message: {stable_message}")

    lines.extend(("", "Upstream error:"))
    if detail.body_text:
        lines.append(detail.body_text)
    else:
        lines.append("(empty upstream error body)")
    if detail.body_truncated and (
        detail.body_text is None
        or f"truncated after {PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES} bytes"
        not in detail.body_text
    ):
        lines.append(
            f"... [truncated after {PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES} bytes]"
        )

    _append_request_id_lines(lines, request_id)
    return "\n".join(lines)


def user_visible_message_for_mapped_provider_error(
    mapped: Exception,
    *,
    provider_name: str,
    read_timeout_s: float | None,
    detail: ProviderErrorDetail | None = None,
    request_id: str | None = None,
) -> str:
    """Return the user-visible string after :func:`map_error` (405 + mapped types)."""
    if detail is not None:
        return format_provider_error_message(
            mapped,
            detail,
            provider_name=provider_name,
            read_timeout_s=read_timeout_s,
            request_id=request_id,
        )
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
        sdk_status = retryable_transient_status(e) or getattr(e, "status_code", None)
        if is_transient_overload_error(e):
            return OverloadedError(message, raw_error=raw_message)
        if isinstance(sdk_status, int) and 500 <= sdk_status <= 599:
            stable = APIError("_", status_code=sdk_status)
            return APIError(
                get_user_facing_error_message(stable),
                status_code=sdk_status,
                raw_error=str(e),
            )
        return APIError(message, status_code=500, raw_error=str(e))
    if isinstance(e, openai.APIError):
        status = retryable_transient_status(e)
        if status == 429:
            limiter.set_blocked(60)
            return RateLimitError(
                "Provider rate limit reached. Please retry shortly.", raw_error=str(e)
            )
        if is_transient_overload_error(e):
            return OverloadedError(
                "Provider is currently overloaded. Please retry.", raw_error=str(e)
            )
        effective_status = status or getattr(e, "status_code", None)
        if not isinstance(effective_status, int):
            effective_status = 500
        stable = APIError("_", status_code=effective_status)
        return APIError(
            get_user_facing_error_message(stable),
            status_code=effective_status,
            raw_error=str(e),
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
