"""Unit tests for Antigravity CLI OAuth authentication module."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.providers.antigravity.auth import (
    ANTIGRAVITY_CLIENT_NAME,
    ANTIGRAVITY_GOOG_API_CLIENT,
    ANTIGRAVITY_USER_AGENT,
    DEFAULT_FALLBACK_PROJECT_ID,
    AntigravityAuth,
    decode_jwt_payload,
    is_token_expired,
    load_antigravity_token,
    load_code_assist_async,
    load_code_assist_sync,
    load_token_from_file,
    parse_expiry,
    refresh_oauth_token_async,
    refresh_oauth_token_sync,
    save_token_to_file,
)
from free_claude_code.providers.exceptions import AuthenticationError


def test_decode_jwt_payload():
    # Valid JWT snippet (header.payload.signature)
    # payload: {"exp": 1754580501, "sub": "12345"}
    # base64url("{"exp": 1754580501, "sub": "12345"}") = eyJleHAiOiAxNzU0NTgwNTAxLCAic3ViIjogIjEyMzQ1In0
    valid_jwt = (
        "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOiAxNzU0NTgwNTAxLCAic3ViIjogIjEyMzQ1In0.signature"
    )
    payload = decode_jwt_payload(valid_jwt)
    assert payload.get("exp") == 1754580501
    assert payload.get("sub") == "12345"

    assert decode_jwt_payload("invalid_token") == {}
    assert decode_jwt_payload("") == {}


def test_parse_expiry():
    # Numeric timestamp
    now_ts = time.time()
    assert parse_expiry(now_ts) == now_ts
    assert parse_expiry(1000) == 1000.0

    # ISO 8601 string with nanoseconds Z
    iso_nano = "2026-08-07T23:28:21.610947673Z"
    parsed_ts = parse_expiry(iso_nano)
    assert parsed_ts is not None
    # 2026-08-07T23:28:21 UTC is 1786145301
    assert parsed_ts > 1700000000

    # ISO string with simple Z
    iso_simple = "2026-08-07T23:28:21Z"
    parsed_simple = parse_expiry(iso_simple)
    assert parsed_simple is not None

    # Invalid / None
    assert parse_expiry(None) is None
    assert parse_expiry("") is None
    assert parse_expiry("not-a-date") is None


def test_is_token_expired():
    future_ts = time.time() + 1000.0
    past_ts = time.time() - 1000.0
    near_future_ts = time.time() + 100.0  # within 300s margin

    assert not is_token_expired(future_ts)
    assert is_token_expired(past_ts)
    assert is_token_expired(near_future_ts, margin_seconds=300)

    # Expiry from JWT payload
    valid_jwt_future = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOiAyMDA0NTgwNTAwfQ.sig"
    valid_jwt_past = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOiAxMDA0NTgwNTAwfQ.sig"
    assert not is_token_expired(None, access_token=valid_jwt_future)
    assert is_token_expired(None, access_token=valid_jwt_past)

    # Empty token without expiry
    assert is_token_expired(None, access_token=None)


def test_load_token_from_file(tmp_path: Path):
    token_file = tmp_path / "antigravity-oauth-token"
    content = {
        "token": {
            "access_token": "test_access_token_123",
            "refresh_token": "test_refresh_token_456",
            "expiry": "2026-08-09T20:00:00Z",
            "token_type": "Bearer",
        },
        "auth_method": "consumer",
    }
    token_file.write_text(json.dumps(content), encoding="utf-8")

    loaded = load_token_from_file(token_file)
    assert loaded is not None
    assert loaded["access_token"] == "test_access_token_123"
    assert loaded["refresh_token"] == "test_refresh_token_456"
    assert loaded["auth_method"] == "consumer"
    assert loaded["file_path"] == str(token_file)

    # Non-existent file
    assert load_token_from_file(tmp_path / "non_existent") is None


def test_load_antigravity_token_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTIGRAVITY_ACCESS_TOKEN", "env_access_token_789")
    monkeypatch.setenv("ANTIGRAVITY_REFRESH_TOKEN", "env_refresh_token_abc")

    token_data = load_antigravity_token()
    assert token_data["access_token"] == "env_access_token_789"
    assert token_data["refresh_token"] == "env_refresh_token_abc"


def test_load_antigravity_token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTIGRAVITY_ACCESS_TOKEN", raising=False)
    token_file = tmp_path / "custom_oauth_token.json"
    token_file.write_text(
        json.dumps(
            {
                "access_token": "custom_file_access_token",
                "refresh_token": "custom_file_refresh_token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTIGRAVITY_TOKEN_FILE", str(token_file))

    token_data = load_antigravity_token()
    assert token_data["access_token"] == "custom_file_access_token"
    assert token_data["refresh_token"] == "custom_file_refresh_token"


def test_load_antigravity_token_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTIGRAVITY_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "free_claude_code.providers.antigravity.auth.find_token_file", lambda: None
    )

    with pytest.raises(AuthenticationError, match="No Antigravity CLI token found"):
        load_antigravity_token()


def test_refresh_oauth_token_sync():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "new_refreshed_access_token",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = refresh_oauth_token_sync("my_refresh_token")
        assert result["access_token"] == "new_refreshed_access_token"
        assert result["expires_in"] == 3600
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "my_refresh_token"


def test_refresh_oauth_token_sync_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"error": "invalid_grant"}'

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(AuthenticationError, match="OAuth token refresh failed"),
    ):
        refresh_oauth_token_sync("bad_refresh_token")


@pytest.mark.asyncio
async def test_refresh_oauth_token_async():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "async_refreshed_access_token",
        "expires_in": 3600,
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp) as mock_post:
        result = await refresh_oauth_token_async("my_refresh_token")
        assert result["access_token"] == "async_refreshed_access_token"
        mock_post.assert_called_once()


def test_save_token_to_file(tmp_path: Path):
    token_file = tmp_path / "save_test_token.json"
    token_data = {
        "access_token": "updated_access_token",
        "refresh_token": "updated_refresh_token",
        "expiry": "2026-08-09T22:00:00Z",
        "token_type": "Bearer",
        "_raw_data": {
            "token": {
                "access_token": "old_token",
                "refresh_token": "old_refresh",
            },
            "auth_method": "consumer",
        },
    }

    assert save_token_to_file(token_file, token_data)
    saved_content = json.loads(token_file.read_text(encoding="utf-8"))
    assert saved_content["token"]["access_token"] == "updated_access_token"
    assert saved_content["auth_method"] == "consumer"


def test_load_code_assist_sync():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "cloudaicompanionProject": "custom-companion-project-999"
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        proj_id = load_code_assist_sync("test_access_token")
        assert proj_id == "custom-companion-project-999"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["User-Agent"] == ANTIGRAVITY_USER_AGENT
        assert kwargs["headers"]["X-Client-Name"] == ANTIGRAVITY_CLIENT_NAME
        assert kwargs["headers"]["X-Goog-Api-Client"] == ANTIGRAVITY_GOOG_API_CLIENT
        assert kwargs["headers"]["Authorization"] == "Bearer test_access_token"
        assert kwargs["json"]["metadata"]["ideType"] == "ANTIGRAVITY"


def test_load_code_assist_sync_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTIGRAVITY_PROJECT_ID", raising=False)
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("httpx.Client.post", return_value=mock_resp):
        proj_id = load_code_assist_sync("test_access_token")
        assert proj_id == DEFAULT_FALLBACK_PROJECT_ID


@pytest.mark.asyncio
async def test_load_code_assist_async():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "cloudaicompanionProject": {"projectId": "nested-project-456"}
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        proj_id = await load_code_assist_async("test_access_token")
        assert proj_id == "nested-project-456"


def test_antigravity_auth_class(tmp_path: Path):
    token_file = tmp_path / "antigravity-oauth-token"
    future_expiry = time.time() + 3600.0
    future_iso = datetime.fromtimestamp(future_expiry, tz=UTC).isoformat()

    content = {
        "token": {
            "access_token": "valid_token_abc",
            "refresh_token": "refresh_token_abc",
            "expiry": future_iso,
            "token_type": "Bearer",
        },
        "auth_method": "consumer",
    }
    token_file.write_text(json.dumps(content), encoding="utf-8")

    auth = AntigravityAuth(token_path=token_file)

    # Valid token (no refresh triggered)
    token = auth.get_access_token()
    assert token == "valid_token_abc"

    # Expired token (triggers refresh)
    past_iso = datetime.fromtimestamp(time.time() - 1000.0, tz=UTC).isoformat()
    content["token"]["expiry"] = past_iso
    token_file.write_text(json.dumps(content), encoding="utf-8")

    mock_refresh_resp = {
        "access_token": "fresh_access_token_xyz",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    with patch(
        "free_claude_code.providers.antigravity.auth.refresh_oauth_token_sync",
        return_value=mock_refresh_resp,
    ):
        token = auth.get_access_token(force_refresh=True)
        assert token == "fresh_access_token_xyz"
