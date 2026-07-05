"""Skip helpers for expected live-smoke environment gaps."""

import httpx
import pytest

from core.anthropic.stream_contracts import SSEEvent, text_content

UPSTREAM_UNAVAILABLE_MARKERS = (
    "connection refused",
    "connecterror",
    "connect timeout",
    "readtimeout",
    "server disconnected",
    "service unavailable",
    "temporary failure",
    "timed out",
    "upstream provider",
)


def is_upstream_unavailable_text(text: str) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in UPSTREAM_UNAVAILABLE_MARKERS)


def skip_upstream_unavailable(reason: str) -> None:
    pytest.skip(f"upstream_unavailable: {reason}")


def fail_missing_env(reason: str) -> None:
    pytest.fail(f"missing_env: {reason}")


def skip_if_upstream_unavailable_exception(exc: Exception) -> None:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.ProxyError,
        ),
    ):
        skip_upstream_unavailable(f"{type(exc).__name__}: {exc}")
    if is_upstream_unavailable_text(f"{type(exc).__name__}: {exc}"):
        skip_upstream_unavailable(f"{type(exc).__name__}: {exc}")


def skip_if_upstream_unavailable_events(events: list[SSEEvent]) -> None:
    text = text_content(events)
    if is_upstream_unavailable_text(text):
        skip_upstream_unavailable(text[:500])

    for event in events:
        if getattr(event, "event", None) != "error":
            continue
        data = getattr(event, "data", {})
        message = ""
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", ""))
            else:
                message = str(data)
        if is_upstream_unavailable_text(message):
            skip_upstream_unavailable(message[:500])
