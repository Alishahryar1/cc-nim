"""Tests for provider error mapping and core error formatting."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest
from httpx import HTTPStatusError, ReadTimeout, Request, Response

from config.constants import PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES
from core.anthropic import (
    append_request_id,
    format_user_error_preview,
    get_user_facing_error_message,
)
from providers.error_mapping import (
    attach_provider_error_body,
    extract_provider_error_detail,
    format_provider_error_message,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)


def _make_openai_error(cls, message="test error", status_code=None):
    """Helper to create openai exceptions with required httpx objects."""
    response = Response(
        status_code=status_code or 500, request=Request("POST", "http://test")
    )
    body = {"error": {"message": message}}
    # openai.APIError base class has a different constructor signature
    if cls is openai.APIError:
        return cls(message, request=Request("POST", "http://test"), body=body)
    return cls(message, response=response, body=body)


class TestMapError:
    """Tests for map_error function."""

    def test_authentication_error(self):
        """openai.AuthenticationError -> AuthenticationError."""
        exc = _make_openai_error(openai.AuthenticationError, status_code=401)
        result = map_error(exc)
        assert isinstance(result, AuthenticationError)
        assert result.status_code == 401

    def test_rate_limit_error(self):
        """openai.RateLimitError -> RateLimitError and triggers global block."""
        exc = _make_openai_error(openai.RateLimitError, status_code=429)
        with patch("providers.error_mapping.GlobalRateLimiter") as mock_rl:
            mock_instance = MagicMock()
            mock_rl.get_instance.return_value = mock_instance
            result = map_error(exc)
            assert isinstance(result, RateLimitError)
            assert result.status_code == 429
            mock_instance.set_blocked.assert_called_once_with(60)

    def test_bad_request_error(self):
        """openai.BadRequestError -> InvalidRequestError."""
        exc = _make_openai_error(openai.BadRequestError, status_code=400)
        result = map_error(exc)
        assert isinstance(result, InvalidRequestError)
        assert result.status_code == 400

    @pytest.mark.parametrize(
        "message",
        ["Server overloaded", "No capacity available"],
        ids=["overloaded", "capacity"],
    )
    def test_internal_server_error_overloaded(self, message):
        """InternalServerError with overloaded/capacity keywords -> OverloadedError."""
        exc = _make_openai_error(
            openai.InternalServerError, message=message, status_code=500
        )
        result = map_error(exc)
        assert isinstance(result, OverloadedError)
        assert result.status_code == 529

    def test_internal_server_error_generic(self):
        """InternalServerError without keywords maps to APIError preserving 5xx."""
        exc = _make_openai_error(
            openai.InternalServerError, message="Unknown error", status_code=500
        )
        result = map_error(exc)
        assert isinstance(result, APIError)
        assert result.status_code == 500

    @pytest.mark.parametrize(
        ("status_code", "expect_substr"),
        [
            (500, "provider api request failed"),
            (502, "temporarily unavailable"),
            (503, "temporarily unavailable"),
            (504, "temporarily unavailable"),
            (599, "provider api request failed"),
        ],
    )
    def test_internal_server_error_preserves_5xx_status_for_messaging(
        self, status_code, expect_substr
    ):
        """InternalServerError carrying HTTP 5xx retains status for stable user messaging."""
        exc = _make_openai_error(
            openai.InternalServerError,
            message=f"upstream {status_code}",
            status_code=status_code,
        )
        result = map_error(exc)
        assert isinstance(result, APIError)
        assert result.status_code == status_code
        assert expect_substr in result.message.lower()

    def test_generic_api_error(self):
        """openai.APIError -> APIError with original status_code."""
        exc = _make_openai_error(
            openai.APIError, message="Bad gateway", status_code=502
        )
        result = map_error(exc)
        assert isinstance(result, APIError)

    def test_unmapped_exception_passthrough(self):
        """Non-openai exceptions are returned as-is."""
        exc = RuntimeError("unexpected")
        result = map_error(exc)
        assert result is exc
        assert isinstance(result, RuntimeError)

    def test_value_error_passthrough(self):
        """ValueError passes through unchanged."""
        exc = ValueError("bad value")
        result = map_error(exc)
        assert result is exc


def test_user_facing_message_read_timeout_empty_string():
    """ReadTimeout wrapping TimeoutError should still produce readable text."""
    timeout_exc = ReadTimeout("")
    message = get_user_facing_error_message(timeout_exc, read_timeout_s=60)
    assert message == "Provider request timed out after 60s."


def test_append_request_id_suffix():
    """Request id suffix should be appended deterministically."""
    message = append_request_id("Provider request failed.", "req_abc123")
    assert message == "Provider request failed. (request_id=req_abc123)"


def test_user_facing_message_bad_request_prefers_mapped_text_over_sdk_string():
    """BadRequestError should map to stable wording even when str(exc) is non-empty."""
    exc = _make_openai_error(
        openai.BadRequestError, message="leaky-upstream-detail", status_code=400
    )
    assert get_user_facing_error_message(exc) == "Invalid request sent to provider."


def test_format_user_error_preview_truncates():
    exc = ValueError("x" * 500)
    preview = format_user_error_preview(exc, max_len=20)
    assert len(preview) == 20
    assert preview == "x" * 20


def test_user_visible_message_for_mapped_provider_error_405():
    mapped = APIError("ignored", status_code=405, raw_error="")
    msg = user_visible_message_for_mapped_provider_error(
        mapped, provider_name="ACME", read_timeout_s=30.0
    )
    assert "ACME" in msg and "405" in msg


def test_openai_bad_request_body_is_user_visible():
    exc = _make_openai_error(
        openai.BadRequestError,
        message="Thinking mode does not support this tool_choice",
        status_code=400,
    )
    mapped = map_error(exc)
    msg = format_provider_error_message(
        mapped,
        extract_provider_error_detail(exc),
        provider_name="NIM",
        read_timeout_s=60.0,
        request_id="req_body",
    )

    assert "Upstream provider NIM returned HTTP 400." in msg
    assert "Category: invalid_request_error" in msg
    assert "Thinking mode does not support this tool_choice" in msg
    assert (
        '{"error":{"message":"Thinking mode does not support this tool_choice"}}' in msg
    )
    assert "Request ID: req_body" in msg


def test_auth_status_with_model_error_body_is_not_only_check_api_key():
    body = {
        "type": "error",
        "error": {
            "type": "ModelError",
            "message": "Model qwen3.7-max is not supported for format oa-compat",
        },
    }
    exc = openai.AuthenticationError(
        "Unauthorized",
        response=Response(status_code=401, request=Request("POST", "http://test")),
        body=body,
    )
    mapped = map_error(exc)
    msg = format_provider_error_message(
        mapped,
        extract_provider_error_detail(exc),
        provider_name="OPENCODE_GO",
        read_timeout_s=60.0,
        request_id="req_model",
    )

    assert "Upstream provider OPENCODE_GO returned HTTP 401." in msg
    assert "Category: ModelError" in msg
    assert "Provider authentication failed. Check API key." in msg
    assert "Model qwen3.7-max is not supported for format oa-compat" in msg
    assert msg != "Provider authentication failed. Check API key."


def test_http_status_error_json_body_is_compact_and_visible():
    response = Response(
        status_code=400,
        request=Request("POST", "http://test"),
        json={"error": {"type": "BadRequest", "message": "bad field"}},
    )
    exc = HTTPStatusError("Bad Request", request=response.request, response=response)
    mapped = map_error(exc)
    msg = user_visible_message_for_mapped_provider_error(
        mapped,
        provider_name="LOCAL",
        read_timeout_s=30.0,
        detail=extract_provider_error_detail(exc),
        request_id="req_json",
    )

    assert "Upstream provider LOCAL returned HTTP 400." in msg
    assert "Category: BadRequest" in msg
    assert '{"error":{"type":"BadRequest","message":"bad field"}}' in msg
    assert "Request ID: req_json" in msg


def test_empty_http_error_body_is_explicitly_reported():
    response = Response(
        status_code=500,
        request=Request("POST", "http://test"),
        content=b"",
    )
    exc = HTTPStatusError("Server Error", request=response.request, response=response)
    mapped = map_error(exc)
    msg = format_provider_error_message(
        mapped,
        extract_provider_error_detail(exc),
        provider_name="EMPTY",
        read_timeout_s=30.0,
    )

    assert "Upstream provider EMPTY returned HTTP 500." in msg
    assert "(empty upstream error body)" in msg


def test_attached_provider_error_body_is_capped_for_display():
    response = Response(
        status_code=500,
        request=Request("POST", "http://test"),
        content=b"",
    )
    exc = HTTPStatusError("Server Error", request=response.request, response=response)
    attach_provider_error_body(exc, "x" * (PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES + 10))
    mapped = map_error(exc)
    msg = format_provider_error_message(
        mapped,
        extract_provider_error_detail(exc),
        provider_name="LONG",
        read_timeout_s=30.0,
    )

    assert f"truncated after {PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES} bytes" in msg
    assert "x" * 100 in msg


def test_streaming_transports_pass_scoped_rate_limiter_to_map_error():
    """Guardrail: streaming adapters must scope reactive 429 handling per provider."""
    root = Path(__file__).resolve().parents[2]
    for path in (
        root / "providers" / "transports" / "anthropic_messages" / "transport.py",
        root / "providers" / "transports" / "openai_chat" / "transport.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "map_error(" in text, str(path)
        assert "rate_limiter=self._global_rate_limiter" in text, str(path)
