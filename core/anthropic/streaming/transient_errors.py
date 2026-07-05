"""Provider-neutral transient upstream error classification."""

import json
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import httpx
import openai

_RATE_LIMIT_MARKERS = frozenset(
    {
        "rate_limit",
        "rate limit",
        "too many requests",
    }
)
_OVERLOAD_MARKERS = frozenset(
    {
        "resourceexhausted",
        "resource exhausted",
        "limit reached",
        "overloaded",
        "capacity",
    }
)
_INTERNAL_ERROR_MARKERS = frozenset(
    {
        "internal_server_error",
        "internal server error",
    }
)


def retryable_transient_status(exc: BaseException) -> int | None:
    """Return an HTTP-like retryable status inferred from an upstream exception."""
    if isinstance(exc, openai.RateLimitError):
        return 429
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status if _is_retryable_status(status) else None

    status = _status_from_exception(exc)
    if _is_retryable_status(status):
        return status

    body = getattr(exc, "body", None)
    body_status = _status_from_body(body)
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
    """Return whether an upstream exception indicates overload/capacity pressure."""
    return _has_marker(transient_error_text(exc), _OVERLOAD_MARKERS)


def transient_error_text(exc: BaseException) -> str:
    """Return normalized exception/body/response text for transient classifiers."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(_body_to_text(body))
    response = getattr(exc, "response", None)
    if response is not None:
        with suppress(Exception):
            parts.append(response.text)
    return " ".join(part for part in parts if part).lower()


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
            parsed = json.loads(body)
        except ValueError:
            return (body,)
        return _body_candidates(parsed)
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
    values = []
    for key in ("type", "code"):
        value = item.get(key)
        if isinstance(value, str):
            values.append(value.lower())
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


def _is_retryable_status(status: int | None) -> bool:
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)
