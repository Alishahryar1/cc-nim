import pytest

from free_claude_code.core.anthropic import (
    anthropic_error_payload,
    anthropic_status_for_error_type,
    get_user_facing_error_message,
    redact_sensitive_error_text,
)


@pytest.mark.parametrize(
    ("error_type", "status_code"),
    [
        ("invalid_request_error", 400),
        ("authentication_error", 401),
        ("billing_error", 402),
        ("permission_error", 403),
        ("not_found_error", 404),
        ("request_too_large", 413),
        ("rate_limit_error", 429),
        ("api_error", 500),
        ("timeout_error", 504),
        ("overloaded_error", 529),
        ("future_error", 500),
    ],
)
def test_anthropic_status_for_error_type(error_type: str, status_code: int) -> None:
    assert anthropic_status_for_error_type(error_type) == status_code


def test_anthropic_error_payload_adds_request_id_and_redacts_credentials() -> None:
    payload = anthropic_error_payload(
        error_type="api_error",
        message="failed token=SECRET authorization: Bearer ALSO_SECRET",
        request_id="req_test",
    )

    assert payload == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "failed token=<redacted> authorization: <redacted>",
        },
        "request_id": "req_test",
    }


def test_unknown_exception_message_is_detailed_but_redacted() -> None:
    error = RuntimeError("gateway failed api_key=SECRET details remain")

    assert get_user_facing_error_message(error) == (
        "gateway failed api_key=<redacted> details remain"
    )


def test_redaction_covers_json_credentials_and_recognizable_key_prefixes() -> None:
    sanitized = redact_sensitive_error_text(
        '{"authorization":"Bearer AUTH_TOKEN","api_key":"sk-live-secret-key",'
        '"client_secret":"CLIENT_SECRET"} raw=nvapi-standalone-secret'
    )

    assert sanitized == (
        '{"authorization":"<redacted>","api_key":"<redacted>",'
        '"client_secret":"<redacted>"} raw=<redacted>'
    )
