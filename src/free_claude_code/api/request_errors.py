"""Shared API request validation and safe error logging."""

from typing import Any, Literal

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    anthropic_error_payload,
    anthropic_error_type_for_failure,
)
from free_claude_code.core.diagnostics import (
    redacted_exception_traceback,
    safe_exception_message,
)
from free_claude_code.core.openai_responses import (
    openai_error_payload,
    openai_error_type_for_failure,
)

WireApi = Literal["messages", "responses"]


def require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


def ordinary_application_error_response(
    error: ApplicationError,
    *,
    wire_api: WireApi,
    request_id: str,
    client_should_retry: bool | None = None,
) -> JSONResponse:
    """Serialize a deterministic application error with explicit retry policy."""
    if wire_api == "responses":
        response = JSONResponse(
            status_code=error.status_code,
            content=openai_error_payload(
                message=error.message,
                error_type=openai_error_type_for_failure(error.kind),
            ),
        )
    else:
        response = JSONResponse(
            status_code=error.status_code,
            content=anthropic_error_payload(
                error_type=anthropic_error_type_for_failure(error.kind),
                message=error.message,
                request_id=request_id,
            ),
        )
    if client_should_retry is not None:
        response.headers["x-should-retry"] = str(client_should_retry).lower()
    return response


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
            logger.error(
                "{} request_id={}: {}",
                context,
                request_id,
                safe_exception_message(exc),
            )
        else:
            logger.error("{}: {}", context, safe_exception_message(exc))
        logger.error(redacted_exception_traceback(exc))
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
        detail=safe_exception_message(exc),
    )
