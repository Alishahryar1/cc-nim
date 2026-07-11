"""Canonical execution failures and their protocol-owned wire mappings."""

import pytest

from free_claude_code.core.anthropic.errors import (
    anthropic_error_type_for_failure,
    anthropic_failure_payload,
    execution_failure_from_anthropic_error,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses.errors import (
    openai_error_type_for_failure,
    openai_failure_payload,
)


@pytest.mark.parametrize(
    ("kind", "anthropic_type", "openai_type"),
    [
        (FailureKind.INVALID_REQUEST, "invalid_request_error", "invalid_request_error"),
        (FailureKind.AUTHENTICATION, "authentication_error", "authentication_error"),
        (FailureKind.PERMISSION, "permission_error", "permission_error"),
        (FailureKind.RATE_LIMIT, "rate_limit_error", "rate_limit_error"),
        (FailureKind.OVERLOADED, "overloaded_error", "overloaded_error"),
        # Existing finalized transport timeouts are exposed as api_error; the
        # semantic kind becomes more precise without changing that wire contract.
        (FailureKind.TIMEOUT, "api_error", "api_error"),
        (FailureKind.UPSTREAM, "api_error", "api_error"),
        (FailureKind.UNAVAILABLE, "api_error", "api_error"),
    ],
)
def test_each_protocol_owns_failure_kind_to_wire_type_mapping(
    kind: FailureKind,
    anthropic_type: str,
    openai_type: str,
) -> None:
    assert anthropic_error_type_for_failure(kind) == anthropic_type
    assert openai_error_type_for_failure(kind) == openai_type


@pytest.mark.parametrize(
    ("native_type", "kind", "status_code"),
    [
        ("billing_error", FailureKind.PERMISSION, 402),
        ("not_found_error", FailureKind.INVALID_REQUEST, 404),
        ("request_too_large", FailureKind.INVALID_REQUEST, 413),
        ("timeout_error", FailureKind.TIMEOUT, 504),
    ],
)
def test_native_error_type_round_trips_through_status_aware_execution_failure(
    native_type: str,
    kind: FailureKind,
    status_code: int,
) -> None:
    failure = execution_failure_from_anthropic_error(
        {
            "type": "error",
            "error": {
                "type": native_type,
                "message": "Native provider failure.",
            },
        }
    )

    assert failure.kind is kind
    assert failure.status_code == status_code
    assert anthropic_failure_payload(failure)["error"]["type"] == native_type
    assert openai_failure_payload(failure)["error"]["type"] == native_type


def test_finalized_status_502_timeout_keeps_existing_api_error_wire_type() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.TIMEOUT,
        status_code=502,
        message="Provider request timed out.",
        retryable=True,
    )

    assert anthropic_failure_payload(failure)["error"]["type"] == "api_error"
    assert openai_failure_payload(failure)["error"]["type"] == "api_error"
