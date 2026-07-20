"""Raw provider failure classification into the canonical neutral model."""

from collections.abc import Callable
from dataclasses import dataclass

import httpx
import openai
import pytest

from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.failure_policy import (
    ProviderRecoveryExhausted,
    classify_provider_failure,
    is_retryable_provider_error,
    retryable_upstream_status,
)


def _openai_status_error(
    error_type: type[openai.APIStatusError],
    *,
    status_code: int,
    message: str,
    body: object | None = None,
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_type(
        message,
        response=response,
        body=body or {"error": {"message": message}},
    )


def _statusless_openai_error(message: str, body: object | None) -> openai.APIError:
    return openai.APIError(
        message,
        request=httpx.Request("POST", "https://provider.test/v1/chat/completions"),
        body=body,
    )


def _http_status_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.test/v1/messages")
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"message": message, "api_key": "SECRET"}},
    )
    return httpx.HTTPStatusError(message, request=request, response=response)


@dataclass(frozen=True, slots=True)
class _ClassificationCase:
    name: str
    error: Callable[[], Exception]
    kind: FailureKind
    status_code: int
    retryable: bool


_CASES = (
    _ClassificationCase(
        "openai_authentication",
        lambda: _openai_status_error(
            openai.AuthenticationError,
            status_code=401,
            message="Unauthorized",
        ),
        FailureKind.AUTHENTICATION,
        401,
        False,
    ),
    _ClassificationCase(
        "openai_rate_limit",
        lambda: _openai_status_error(
            openai.RateLimitError,
            status_code=429,
            message="Too many requests",
        ),
        FailureKind.RATE_LIMIT,
        429,
        True,
    ),
    _ClassificationCase(
        "openai_bad_request",
        lambda: _openai_status_error(
            openai.BadRequestError,
            status_code=400,
            message="bad tool shape",
        ),
        FailureKind.INVALID_REQUEST,
        400,
        False,
    ),
    _ClassificationCase(
        "openai_overload_marker",
        lambda: _openai_status_error(
            openai.InternalServerError,
            status_code=500,
            message="No capacity available",
        ),
        FailureKind.OVERLOADED,
        529,
        True,
    ),
    _ClassificationCase(
        "openai_generic_503_preserved",
        lambda: _openai_status_error(
            openai.InternalServerError,
            status_code=503,
            message="generic server failure",
        ),
        FailureKind.UPSTREAM,
        503,
        True,
    ),
    _ClassificationCase(
        "statusless_openai_rate_limit_body",
        lambda: _statusless_openai_error(
            "stream embedded error",
            {"error": {"message": "too many requests", "code": 429}},
        ),
        FailureKind.RATE_LIMIT,
        429,
        True,
    ),
    _ClassificationCase(
        "statusless_openai_overload_body",
        lambda: _statusless_openai_error(
            "ResourceExhausted: limit reached",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
        FailureKind.OVERLOADED,
        529,
        True,
    ),
    _ClassificationCase(
        "statusless_openai_unknown_is_not_retryable",
        lambda: _statusless_openai_error(
            "stream embedded error",
            {"error": {"message": "unknown provider failure"}},
        ),
        FailureKind.UPSTREAM,
        500,
        False,
    ),
    _ClassificationCase(
        "http_403_keeps_authentication_quirk",
        lambda: _http_status_error(403, "Forbidden"),
        FailureKind.AUTHENTICATION,
        401,
        False,
    ),
    _ClassificationCase(
        "http_502_keeps_overload_quirk",
        lambda: _http_status_error(502, "Bad gateway"),
        FailureKind.OVERLOADED,
        529,
        True,
    ),
    _ClassificationCase(
        "http_599_preserves_status",
        lambda: _http_status_error(599, "Upstream failure"),
        FailureKind.UPSTREAM,
        599,
        True,
    ),
    _ClassificationCase(
        "http_405_is_not_retryable",
        lambda: _http_status_error(405, "Wrong endpoint"),
        FailureKind.UPSTREAM,
        405,
        False,
    ),
    _ClassificationCase(
        "read_timeout_keeps_pre_start_status",
        lambda: httpx.ReadTimeout(
            "",
            request=httpx.Request("POST", "https://provider.test/v1/messages"),
        ),
        FailureKind.TIMEOUT,
        502,
        True,
    ),
    _ClassificationCase(
        "openai_connection_error_keeps_status",
        lambda: openai.APIConnectionError(
            request=httpx.Request("POST", "https://provider.test/v1/chat/completions")
        ),
        FailureKind.UNAVAILABLE,
        500,
        True,
    ),
    _ClassificationCase(
        "unknown_exception_keeps_gateway_status",
        lambda: RuntimeError("unexpected provider failure"),
        FailureKind.UPSTREAM,
        502,
        False,
    ),
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_raw_provider_failure_maps_to_canonical_failure(
    case: _ClassificationCase,
) -> None:
    failure = classify_provider_failure(
        case.error(),
        provider_name="TEST_PROVIDER",
        read_timeout_s=30.0,
        request_id="req_classification",
    )

    assert isinstance(failure, ExecutionFailure)
    assert failure.kind is case.kind
    assert failure.status_code == case.status_code
    assert failure.retryable is case.retryable
    assert failure.message.strip()
    assert "Request ID: req_classification" in failure.message
    assert "SECRET" not in failure.message


def test_classification_preserves_useful_body_while_redacting_credentials() -> None:
    error = _http_status_error(
        400,
        "unsupported model format authorization: Bearer AUTH_SECRET",
    )

    failure = classify_provider_failure(
        error,
        provider_name="LOCAL",
        read_timeout_s=60.0,
        request_id="req_body",
    )

    assert failure.kind is FailureKind.INVALID_REQUEST
    assert failure.status_code == 400
    assert "Upstream provider LOCAL returned HTTP 400." in failure.message
    assert "unsupported model format" in failure.message
    assert "Request ID: req_body" in failure.message
    assert "AUTH_SECRET" not in failure.message
    assert "SECRET" not in failure.message


def test_auth_failure_preserves_model_error_body_instead_of_masking_it() -> None:
    error = _openai_status_error(
        openai.AuthenticationError,
        status_code=401,
        message="Unauthorized",
        body={
            "type": "error",
            "error": {
                "type": "ModelError",
                "message": ("Model qwen3.7-max is not supported for format oa-compat"),
            },
        },
    )

    failure = classify_provider_failure(
        error,
        provider_name="OPENCODE_GO",
        read_timeout_s=60.0,
        request_id="req_model",
    )

    assert failure.kind is FailureKind.AUTHENTICATION
    assert failure.status_code == 401
    assert "Category: ModelError" in failure.message
    assert "Provider authentication failed. Check API key." in failure.message
    assert "Model qwen3.7-max is not supported for format oa-compat" in failure.message
    assert "Request ID: req_model" in failure.message


def test_empty_http_error_body_is_reported_explicitly() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/messages")
    response = httpx.Response(500, request=request, content=b"")
    error = httpx.HTTPStatusError(
        "Server Error",
        request=request,
        response=response,
    )

    failure = classify_provider_failure(
        error,
        provider_name="EMPTY",
        read_timeout_s=30.0,
        request_id="req_empty",
    )

    assert failure.kind is FailureKind.UPSTREAM
    assert failure.status_code == 500
    assert "Upstream provider EMPTY returned HTTP 500." in failure.message
    assert "(empty upstream error body)" in failure.message


def test_http_405_diagnostic_names_rejected_upstream_endpoint() -> None:
    failure = classify_provider_failure(
        _http_status_error(405, "Method Not Allowed"),
        provider_name="LOCAL",
        read_timeout_s=30.0,
        request_id="req_405",
    )

    assert failure.kind is FailureKind.UPSTREAM
    assert failure.status_code == 405
    assert (
        "Upstream provider LOCAL rejected the request method or endpoint (HTTP 405)."
        in failure.message
    )
    assert "Request ID: req_405" in failure.message


def test_connection_cause_chain_is_redacted_and_capped() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    error.__cause__ = httpx.ConnectError(
        "connect failed authorization: Bearer CAUSE_SECRET "
        + "x" * (ERROR_DETAIL_DISPLAY_CAP_BYTES + 10),
        request=request,
    )

    failure = classify_provider_failure(
        error,
        provider_name="NIM",
        read_timeout_s=30.0,
        request_id="req_cause",
    )

    assert "Caused by:" in failure.message
    assert "ConnectError: connect failed authorization: <redacted>" in failure.message
    assert "CAUSE_SECRET" not in failure.message
    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in failure.message
    assert "Request ID: req_cause" in failure.message


def test_attached_streamed_error_body_remains_bounded() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/messages")
    response = httpx.Response(500, request=request, content=b"")
    error = httpx.HTTPStatusError(
        "Server Error",
        request=request,
        response=response,
    )
    attach_upstream_error_body(
        error,
        "x" * (ERROR_DETAIL_DISPLAY_CAP_BYTES + 10),
    )

    failure = classify_provider_failure(
        error,
        provider_name="LONG",
        read_timeout_s=30.0,
        request_id="req_long",
    )

    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in failure.message
    assert "x" * 100 in failure.message


def test_shared_recovery_exhaustion_preserves_last_provider_failure() -> None:
    raw_error = _http_status_error(503, "provider still unavailable")
    exhausted = ProviderRecoveryExhausted(raw_error)

    failure = classify_provider_failure(
        exhausted,
        provider_name="SHARED",
        read_timeout_s=30.0,
        request_id="req_exhausted",
    )

    assert failure.kind is FailureKind.OVERLOADED
    assert failure.status_code == 529
    assert "provider still unavailable" in failure.message
    assert "Request ID: req_exhausted" in failure.message
    assert retryable_upstream_status(exhausted) is None
    assert not is_retryable_provider_error(exhausted)
