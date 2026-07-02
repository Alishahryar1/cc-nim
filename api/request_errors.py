"""Shared API request validation and safe error logging."""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import HTTPException
from loguru import logger

from config.settings import Settings
from core.anthropic import get_user_facing_error_message
from providers.exceptions import InvalidRequestError


def require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


def http_status_for_unexpected_api_exception(_exc: BaseException) -> int:
    return 500


def log_unexpected_api_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log API failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def unexpected_http_exception(
    settings: Settings, exc: Exception, *, context: str
) -> HTTPException:
    log_unexpected_api_exception(settings, exc, context=context)
    return HTTPException(
        status_code=http_status_for_unexpected_api_exception(exc),
        detail=get_user_facing_error_message(exc),
    )
