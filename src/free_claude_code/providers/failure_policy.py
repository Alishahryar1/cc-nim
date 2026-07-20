"""Provider-owned SDK classification and retry qualification."""

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any

import httpx
import openai

from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    format_execution_failure_message,
    safe_exception_message,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind

ProviderFailureOverride = Callable[[Exception], ExecutionFailure | None]

_RATE_LIMIT_MARKERS = frozenset({"rate_limit", "rate limit", "too many requests"})
_OVERLOAD_MARKERS = frozenset(
    {
        "resourceexhausted",
        "resource exhausted",
        "limit reached",
        "overloaded",
        "capacity",
    }
)
_INTERNAL_ERROR_MARKERS = frozenset({"internal_server_error", "internal server error"})
_AUTHENTICATION_MESSAGE = "Provider authentication failed. Check API key."
_RATE_LIMIT_MESSAGE = "Provider rate limit reached. Please retry shortly."
_INVALID_REQUEST_MESSAGE = "Invalid request sent to provider."
_CONTEXT_WINDOW_EXCEEDED_MESSAGE = "Provider input exceeds the model context window."
_OVERLOADED_MESSAGE = "Provider is currently overloaded. Please retry."


class ProviderRecoveryExhausted(RuntimeError):
    """A shared provider recovery episode exhausted its single probe budget."""

    def __init__(self, last_error: Exception) -> None:
        super().__init__("Provider recovery retry budget was exhausted.")
        self.last_error = last_error


class RetryableProviderProtocolError(RuntimeError):
    """A malformed upstream protocol result eligible for provider retry."""


def classify_provider_failure(
    exc: Exception,
    *,
    provider_name: str,
    read_timeout_s: float | None,
    request_id: str | None,
    provider_failure_override: ProviderFailureOverride | None = None,
) -> ExecutionFailure:
    """Return one detailed canonical failure after provider retries are exhausted."""
    exc = underlying_provider_error(exc)
    if isinstance(exc, ExecutionFailure):
        failure = exc
        message = failure.message
        request_id_line = f"Request ID: {request_id}" if request_id else None
        if request_id_line and request_id_line not in message:
            message = f"{message}\n\n{request_id_line}"
        return replace(failure, message=message)

    failure = (
        provider_failure_override(exc)
        if provider_failure_override is not None
        else None
    )
    if failure is None:
        failure = _classify_provider_failure(
            exc,
            read_timeout_s=read_timeout_s,
        )
    message = format_execution_failure_message(
        failure,
        extract_upstream_error_detail(exc),
        upstream_name=provider_name,
        request_id=request_id,
    )
    return replace(failure, message=message)


def overloaded_provider_failure() -> ExecutionFailure:
    """Return the canonical provider-overload meaning and stable wording."""
    return _failure(FailureKind.OVERLOADED, 529, _OVERLOADED_MESSAGE, True)


def context_window_exceeded_provider_failure(
    message: str = _CONTEXT_WINDOW_EXCEEDED_MESSAGE,
) -> ExecutionFailure:
    """Return the canonical non-retryable provider context-window failure."""
    return _failure(FailureKind.CONTEXT_WINDOW_EXCEEDED, 400, message, False)


def retryable_transient_status(exc: BaseException) -> int | None:
    """Infer a retryable HTTP-like status from one upstream exception."""
    if isinstance(exc, ProviderRecoveryExhausted):
        return None
    if isinstance(exc, ExecutionFailure):
        status = exc.status_code
        return status if exc.retryable and _is_retryable_status(status) else None
    if isinstance(exc, openai.RateLimitError):
        return 429
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status if _is_retryable_status(status) else None

    status = _status_from_exception(exc)
    if _is_retryable_status(status):
        return status

    body_status = _status_from_body(getattr(exc, "body", None))
    if _is_retryable_status(body_status):
        return body_status

    text = transient_error_text(exc)
    if _has_marker(text, _RATE_LIMIT_MARKERS):
        return 429
    if _has_marker(text, _OVERLOAD_MARKERS):
        return 503
    if _has_marker(text, _INTERNAL_ERROR_MARKERS):
        return 500
    return None


def is_transient_overload_error(exc: BaseException) -> bool:
    """Return whether an upstream exception reports overload or capacity pressure."""
    if isinstance(exc, ProviderRecoveryExhausted):
        return False
    if isinstance(exc, ExecutionFailure):
        return exc.kind == FailureKind.OVERLOADED
    return _has_marker(transient_error_text(exc), _OVERLOAD_MARKERS)


def transient_error_text(exc: BaseException) -> str:
    """Combine exception, body, and response text for provider classification."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(_body_to_text(body))
    response = getattr(exc, "response", None)
    if response is not None:
        with suppress(Exception):
            parts.append(response.text)
    return " ".join(part for part in parts if part).lower()


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Return whether provider policy permits stream retry or recovery."""
    if isinstance(exc, ProviderRecoveryExhausted):
        return False
    if isinstance(exc, ExecutionFailure):
        return exc.retryable
    if isinstance(exc, openai.AuthenticationError | openai.BadRequestError):
        return False
    if retryable_transient_status(exc) is not None:
        return True
    return isinstance(
        exc,
        (
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            RetryableProviderProtocolError,
        ),
    )


def retryable_upstream_status(exc: BaseException) -> int | None:
    """Return a status eligible for provider-opening backoff."""
    status = retryable_transient_status(exc)
    return status if status is not None and _is_retryable_status(status) else None


def provider_error_message(
    exc: BaseException,
    *,
    read_timeout_s: float | None = None,
) -> str:
    """Map raw provider exception types to stable customer-facing wording."""
    if isinstance(exc, Exception):
        exc = underlying_provider_error(exc)
    if isinstance(exc, ExecutionFailure):
        return exc.message
    if isinstance(exc, httpx.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Provider request timed out."
    if isinstance(exc, httpx.ConnectTimeout | httpx.ConnectError):
        return "Could not connect to provider."
    if isinstance(exc, httpx.RemoteProtocolError):
        return "Provider connection was interrupted before a response was received."
    if isinstance(exc, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."
    if isinstance(exc, openai.RateLimitError):
        return _RATE_LIMIT_MESSAGE
    if isinstance(exc, openai.AuthenticationError):
        return _AUTHENTICATION_MESSAGE
    if isinstance(exc, openai.BadRequestError):
        return _INVALID_REQUEST_MESSAGE
    return safe_exception_message(exc)


def _classify_provider_failure(
    exc: Exception,
    *,
    read_timeout_s: float | None,
) -> ExecutionFailure:
    if isinstance(exc, ExecutionFailure):
        return exc

    if isinstance(exc, openai.AuthenticationError):
        return _failure(FailureKind.AUTHENTICATION, 401, _AUTHENTICATION_MESSAGE, False)
    if isinstance(exc, openai.RateLimitError):
        return _failure(FailureKind.RATE_LIMIT, 429, _RATE_LIMIT_MESSAGE, True)
    if isinstance(exc, openai.BadRequestError):
        return _failure(
            FailureKind.INVALID_REQUEST, 400, _INVALID_REQUEST_MESSAGE, False
        )
    if isinstance(exc, openai.APITimeoutError):
        return _failure(FailureKind.TIMEOUT, 500, _stable_upstream(500), True)
    if isinstance(exc, openai.APIConnectionError):
        return _failure(FailureKind.UNAVAILABLE, 500, _stable_upstream(500), True)
    if isinstance(exc, openai.InternalServerError):
        status = retryable_transient_status(exc) or getattr(exc, "status_code", None)
        if is_transient_overload_error(exc):
            return overloaded_provider_failure()
        if isinstance(status, int) and 500 <= status <= 599:
            return _failure(
                FailureKind.UPSTREAM,
                status,
                _stable_upstream(status),
                True,
            )
        return _failure(FailureKind.UPSTREAM, 500, _stable_upstream(500), True)
    if isinstance(exc, openai.APIError):
        status = retryable_transient_status(exc)
        if status == 429:
            return _failure(FailureKind.RATE_LIMIT, 429, _RATE_LIMIT_MESSAGE, True)
        if is_transient_overload_error(exc):
            return overloaded_provider_failure()
        effective_status = status or getattr(exc, "status_code", None)
        if not isinstance(effective_status, int):
            effective_status = 500
        return _failure(
            FailureKind.UPSTREAM,
            effective_status,
            _stable_upstream(effective_status),
            is_retryable_provider_error(exc),
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return _failure(
                FailureKind.AUTHENTICATION, 401, _AUTHENTICATION_MESSAGE, False
            )
        if status == 429:
            return _failure(FailureKind.RATE_LIMIT, 429, _RATE_LIMIT_MESSAGE, True)
        if status == 400:
            return _failure(
                FailureKind.INVALID_REQUEST, 400, _INVALID_REQUEST_MESSAGE, False
            )
        if status in (502, 503, 504):
            return overloaded_provider_failure()
        return _failure(
            FailureKind.UPSTREAM,
            status,
            _stable_upstream(status),
            _is_retryable_status(status),
        )

    kind = FailureKind.UPSTREAM
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        kind = FailureKind.TIMEOUT
    elif isinstance(exc, httpx.ConnectError | httpx.NetworkError):
        kind = FailureKind.UNAVAILABLE
    return _failure(
        kind,
        502,
        provider_error_message(exc, read_timeout_s=read_timeout_s),
        is_retryable_provider_error(exc),
    )


def _failure(
    kind: FailureKind,
    status_code: int,
    message: str,
    retryable: bool,
) -> ExecutionFailure:
    return ExecutionFailure(
        kind=kind,
        status_code=status_code,
        message=message,
        retryable=retryable,
    )


def _stable_upstream(status_code: int) -> str:
    if status_code in (502, 503, 504):
        return "Provider is temporarily unavailable. Please retry."
    return "Provider API request failed."


def _status_from_exception(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _status_from_body(body: Any) -> int | None:
    for item in _body_candidates(body):
        if not isinstance(item, Mapping):
            continue
        for key in ("status", "status_code", "code"):
            status = _coerce_status(item.get(key))
            if status is not None:
                return status
        type_status = _status_from_type_fields(item)
        if type_status is not None:
            return type_status
    return None


def _body_candidates(body: Any) -> tuple[Any, ...]:
    if isinstance(body, str):
        try:
            return _body_candidates(json.loads(body))
        except ValueError:
            return (body,)
    if isinstance(body, bytes):
        return _body_candidates(body.decode("utf-8", errors="replace"))
    if isinstance(body, Mapping):
        nested = body.get("error")
        return (body, nested) if isinstance(nested, Mapping) else (body,)
    return (body,)


def _coerce_status(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _status_from_type_fields(item: Mapping[str, Any]) -> int | None:
    values = [
        value.lower()
        for key in ("type", "code")
        if isinstance((value := item.get(key)), str)
    ]
    text = " ".join(values)
    if _has_marker(text, _RATE_LIMIT_MARKERS):
        return 429
    if _has_marker(text, _OVERLOAD_MARKERS):
        return 503
    if _has_marker(text, _INTERNAL_ERROR_MARKERS):
        return 500
    return None


def _body_to_text(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(body)


def _has_marker(text: str, markers: frozenset[str]) -> bool:
    return any(marker in text for marker in markers)


def underlying_provider_error(exc: Exception) -> Exception:
    """Return the raw failure retained by an exhausted recovery wrapper."""
    while isinstance(exc, ProviderRecoveryExhausted):
        exc = exc.last_error
    return exc


def _is_retryable_status(status: int | None) -> bool:
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)
