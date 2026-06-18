"""HTTP helpers for native Anthropic Messages transports."""

from __future__ import annotations

import inspect
from typing import Any

import httpx
from loguru import logger

from config.constants import (
    NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES,
    PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES,
)
from providers.error_mapping import attach_provider_error_body
from providers.exceptions import ModelListResponseError


async def maybe_await_aclose(response: Any) -> None:
    """Call ``aclose`` on httpx-like responses; ignore sync test doubles."""
    close = getattr(response, "aclose", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def model_list_json(response: httpx.Response, *, provider_name: str) -> Any:
    """Parse model-list JSON with a provider-specific malformed-body error."""
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ModelListResponseError(
            f"{provider_name} model-list response is malformed: invalid JSON"
        ) from exc


async def read_error_body_preview(
    response: httpx.Response, max_bytes: int
) -> tuple[bytes, bool]:
    """Read at most ``max_bytes`` from an error response body."""
    if max_bytes <= 0:
        return b"", False
    received = 0
    parts: list[bytes] = []
    truncated = False
    async for chunk in response.aiter_bytes(chunk_size=65_536):
        if received >= max_bytes:
            truncated = True
            break
        remaining = max_bytes - received
        take = chunk if len(chunk) <= remaining else chunk[:remaining]
        if take:
            parts.append(take)
        received += len(take)
        if len(chunk) > len(take):
            truncated = True
            break
        if received >= max_bytes:
            break
    return (b"".join(parts), truncated)


async def raise_for_status_with_body(
    response: httpx.Response,
    *,
    provider_name: str,
    req_tag: str,
    log_api_error_tracebacks: bool,
) -> None:
    """Raise for non-200 responses after attaching a safe body preview."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        preview, truncated = await read_error_body_preview(
            response, PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES
        )
        attach_provider_error_body(error, preview, truncated=truncated)
        if log_api_error_tracebacks:
            log_preview = preview[:NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES]
            log_truncated = truncated or len(preview) > len(log_preview)
            if log_preview:
                text = log_preview.decode("utf-8", errors="replace")
                logger.error(
                    "{}_ERROR:{} HTTP {} body_preview_bytes={} truncated={}: {}",
                    provider_name,
                    req_tag,
                    response.status_code,
                    len(log_preview),
                    log_truncated,
                    text,
                )
            else:
                logger.error(
                    "{}_ERROR:{} HTTP {} (empty error body)",
                    provider_name,
                    req_tag,
                    response.status_code,
                )
        else:
            cl = response.headers.get("content-length", "").strip()
            extra = f" content_length_declared={cl}" if cl.isdigit() else ""
            body_extra = (
                " empty_error_body"
                if not preview
                else f" error_body_bytes_read={len(preview)}"
            )
            logger.error(
                "{}_ERROR:{} HTTP {}{}{}",
                provider_name,
                req_tag,
                response.status_code,
                extra,
                body_extra,
            )
        raise error
