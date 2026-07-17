"""Tests for the direct ChatGPT OAuth Responses API provider."""

import json
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.errors import (
    InvalidRequestError,
)
from free_claude_code.config.provider_catalog import CHATGPT_OAUTH_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.chatgpt_oauth import ChatGPTOAuthProvider
from free_claude_code.providers.chatgpt_oauth.conversion import (
    build_chatgpt_oauth_request_body,
    chatgpt_tool_call_to_anthropic,
)
from free_claude_code.providers.chatgpt_oauth.credentials import (
    load_chatgpt_oauth_credentials,
)
from free_claude_code.providers.chatgpt_oauth.streaming import (
    ChatGPTOAuthStreamConverter,
)
from tests.providers.support import passthrough_rate_limiter


def _provider_config(**overrides) -> ProviderConfig:
    return ProviderConfig(
        api_key=overrides.get("api_key", "test_token"),
        base_url=overrides.get("base_url", CHATGPT_OAUTH_DEFAULT_BASE),
        rate_limit=overrides.get("rate_limit", 10),
        rate_window=overrides.get("rate_window", 60),
        max_concurrency=overrides.get("max_concurrency", 5),
        http_read_timeout=overrides.get("http_read_timeout", 300.0),
        http_write_timeout=overrides.get("http_write_timeout", 10.0),
        http_connect_timeout=overrides.get("http_connect_timeout", 60.0),
        proxy=overrides.get("proxy", ""),
    )


@pytest.fixture
def chatgpt_oauth_provider():
    return ChatGPTOAuthProvider(
        _provider_config(),
        rate_limiter=passthrough_rate_limiter(),
        account_id="test_account_id",
    )


def test_provider_uses_default_base_url():
    provider = ChatGPTOAuthProvider(
        _provider_config(base_url=""),
        rate_limiter=passthrough_rate_limiter(),
    )
    assert provider._base_url == CHATGPT_OAUTH_DEFAULT_BASE


def test_provider_uses_configured_base_url():
    provider = ChatGPTOAuthProvider(
        _provider_config(base_url="https://example.com/backend-api"),
        rate_limiter=passthrough_rate_limiter(),
    )
    assert provider._base_url == "https://example.com/backend-api"


def test_build_request_body_converts_messages():
    request = MessagesRequest(
        model="gpt-5",
        max_tokens=50,
        messages=[Message(role="user", content="hi")],
    )

    body = build_chatgpt_oauth_request_body(request)

    assert body["model"] == "gpt-5"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["parallel_tool_calls"] is False
    assert "max_output_tokens" not in body
    assert body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]


def test_build_request_body_adds_reasoning_for_gpt5():
    request = MessagesRequest(
        model="gpt-5",
        messages=[Message(role="user", content="hi")],
    )

    body = build_chatgpt_oauth_request_body(request)

    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "reasoning.encrypted_content" in body["include"]


def test_build_request_body_skips_reasoning_for_non_reasoning_model():
    request = MessagesRequest(
        model="unknown-model",
        messages=[Message(role="user", content="hi")],
    )

    body = build_chatgpt_oauth_request_body(request)

    assert "reasoning" not in body


def test_build_request_body_extracts_system_as_instructions():
    request = MessagesRequest(
        model="gpt-5",
        system="You are a helpful assistant.",
        messages=[Message(role="user", content="hi")],
    )

    body = build_chatgpt_oauth_request_body(request)

    assert body["instructions"] == "You are a helpful assistant."


def test_build_request_body_rejects_extra_body():
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"x": 1},
        }
    )

    with pytest.raises(InvalidRequestError):
        build_chatgpt_oauth_request_body(request)


def test_chatgpt_tool_call_to_anthropic():
    item = {
        "id": "call_1",
        "name": "bash",
        "arguments": json.dumps({"command": "ls"}),
    }

    block = chatgpt_tool_call_to_anthropic(item)

    assert block["type"] == "tool_use"
    assert block["id"] == "call_1"
    assert block["name"] == "bash"
    assert block["input"] == {"command": "ls"}


def test_stream_converter_emits_text_delta():
    from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger

    ledger = AnthropicStreamLedger("msg_1", "gpt-5", input_tokens=0)
    converter = ChatGPTOAuthStreamConverter(ledger)

    events = list(converter.feed({"type": "response.output_text.delta", "delta": "hello"}))

    assert any("content_block_start" in e and "text" in e for e in events)
    assert any("text_delta" in e and "hello" in e for e in events)


def test_stream_converter_emits_tool_call():
    from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger

    ledger = AnthropicStreamLedger("msg_1", "gpt-5", input_tokens=0)
    converter = ChatGPTOAuthStreamConverter(ledger)

    events = list(converter.feed({
        "type": "response.output_item.added",
        "item": {"type": "function_call", "id": "call_1", "name": "bash"},
    }))
    events += list(converter.feed({
        "type": "response.function_call_arguments.delta",
        "item_id": "call_1",
        "delta": '{"command":',
    }))
    events += list(converter.feed({
        "type": "response.function_call_arguments.delta",
        "item_id": "call_1",
        "delta": '"ls"}',
    }))
    events += list(converter.feed({
        "type": "response.output_item.done",
        "item": {"type": "function_call", "id": "call_1", "arguments": '{"command":"ls"}'},
    }))

    assert any("tool_use" in e for e in events)
    assert any('"bash"' in e for e in events)
    # partial_json contains the raw argument string; look for the key without
    # requiring exact JSON quote escaping in the serialized SSE line.
    assert any("command" in e for e in events)


@pytest.mark.asyncio
async def test_list_model_ids_returns_known_models(chatgpt_oauth_provider):
    models = await chatgpt_oauth_provider.list_model_ids()
    # OpenCode-aligned allowlist: explicit allows, disallowed pro, and a
    # version heuristic that keeps GPT-5.x models newer than 5.4.
    assert "gpt-5.5" in models
    assert "gpt-5.4" in models
    assert "gpt-5.4-mini" in models
    assert "gpt-5.3-codex-spark" in models
    assert "gpt-5.5-pro" not in models
    assert "gpt-5.6" not in models
    assert "gpt-5.2-codex" not in models


@pytest.mark.asyncio
async def test_cleanup_closes_http_client(chatgpt_oauth_provider):
    chatgpt_oauth_provider._client.aclose = AsyncMock()

    await chatgpt_oauth_provider.cleanup()

    chatgpt_oauth_provider._client.aclose.assert_awaited_once()


def test_load_credentials_prefers_explicit_token(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"tokens": {"access_token": "file_token"}}', encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    creds = load_chatgpt_oauth_credentials(
        access_token="explicit_token",
        account_id="explicit_account",
    )

    assert creds.access_token == "explicit_token"
    assert creds.account_id == "explicit_account"


def test_load_credentials_reads_codex_auth_file(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"tokens": {"access_token": "file_token"}}', encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    creds = load_chatgpt_oauth_credentials()

    assert creds.access_token == "file_token"


def _jwt(payload_dict: dict) -> str:
    import base64

    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(payload_dict).encode()).decode().rstrip("=")
    return f"{header}.{payload}."


def test_load_credentials_prefers_id_token_for_account_id(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    access_token = _jwt({"exp": 9999999999})
    id_token = _jwt({"chatgpt_account_id": "acct_from_id_token"})
    (codex_home / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": access_token, "id_token": id_token}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    creds = load_chatgpt_oauth_credentials()

    assert creds.account_id == "acct_from_id_token"


def test_load_credentials_falls_back_to_organization_id(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    access_token = _jwt({"organizations": [{"id": "org_123"}], "exp": 9999999999})
    (codex_home / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": access_token}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    creds = load_chatgpt_oauth_credentials()

    assert creds.account_id == "org_123"


def test_refresh_persists_rotated_tokens_to_auth_file(tmp_path, monkeypatch):
    import time

    from free_claude_code.providers.chatgpt_oauth import credentials as creds_module

    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    expired_access = _jwt({"exp": int(time.time()) - 100})
    auth_path.write_text(
        json.dumps(
            {"tokens": {"access_token": expired_access, "refresh_token": "refresh_old"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    new_access = _jwt({"exp": 9999999999, "chatgpt_account_id": "acct_new"})
    monkeypatch.setattr(
        creds_module,
        "_refresh_access_token",
        lambda refresh: (new_access, "refresh_rotated", 9999999999, None),
    )

    creds = load_chatgpt_oauth_credentials()

    assert creds.access_token == new_access
    assert creds.account_id == "acct_new"
    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == new_access
    assert saved["tokens"]["refresh_token"] == "refresh_rotated"
    assert saved["tokens"]["expires_at"] == 9999999999


def test_write_codex_auth_file_stores_id_token(tmp_path, monkeypatch):
    from free_claude_code.providers.chatgpt_oauth import oauth_login

    auth_path = tmp_path / ".codex" / "auth.json"
    oauth_login._write_codex_auth_file(
        {
            "access_token": "access_1",
            "refresh_token": "refresh_1",
            "id_token": "id_1",
            "expires_in": 3600,
        },
        auth_path=auth_path,
    )

    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["tokens"]["id_token"] == "id_1"


def test_load_credentials_extracts_account_id_from_jwt(tmp_path, monkeypatch):
    import base64

    header = base64.urlsafe_b64encode(b'{}').decode().rstrip("=")
    payload_dict = {
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_dict).encode()
    ).decode().rstrip("=")
    token = f"{header}.{payload}."

    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": token}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    creds = load_chatgpt_oauth_credentials()

    assert creds.account_id == "acct_123"


def test_build_headers_matches_opencode_shape(chatgpt_oauth_provider):
    from free_claude_code.providers.chatgpt_oauth.credentials import (
        ChatGPTOAuthCredentials,
    )
    from free_claude_code.providers.chatgpt_oauth.provider import _build_headers

    credentials = ChatGPTOAuthCredentials(
        access_token="test_token",
        account_id="acct_123",
    )
    headers = _build_headers(credentials, chatgpt_oauth_provider._session_id)

    assert headers["Authorization"] == "Bearer test_token"
    assert headers["originator"] == "free-claude-code"
    assert headers["session-id"] == chatgpt_oauth_provider._session_id
    assert "User-Agent" in headers
    assert headers["User-Agent"].startswith("free-claude-code/")
    assert headers["ChatGPT-Account-ID"] == "acct_123"


def test_session_id_is_stable_per_provider():
    from free_claude_code.providers.base import ProviderConfig

    config = ProviderConfig(
        api_key="x",
        base_url="",
        rate_limit=10,
        rate_window=60,
        max_concurrency=5,
        http_read_timeout=300.0,
        http_write_timeout=10.0,
        http_connect_timeout=60.0,
        proxy="",
    )
    provider1 = ChatGPTOAuthProvider(
        config,
        rate_limiter=passthrough_rate_limiter(),
    )
    provider2 = ChatGPTOAuthProvider(
        config,
        rate_limiter=passthrough_rate_limiter(),
    )
    assert provider1._session_id
    assert provider1._session_id == provider1._session_id
    assert provider1._session_id != provider2._session_id


def test_model_filter_logic():
    from free_claude_code.providers.chatgpt_oauth.provider import (
        _is_chatgpt_oauth_model,
    )

    assert _is_chatgpt_oauth_model("gpt-5.5") is True
    assert _is_chatgpt_oauth_model("gpt-5.4") is True
    assert _is_chatgpt_oauth_model("gpt-5.4-mini") is True
    assert _is_chatgpt_oauth_model("gpt-5.7") is True
    assert _is_chatgpt_oauth_model("gpt-5.5-pro") is False
    assert _is_chatgpt_oauth_model("gpt-5.6") is False
    assert _is_chatgpt_oauth_model("gpt-5.2-codex") is False
    assert _is_chatgpt_oauth_model("codex-mini-latest") is False


def test_perform_chatgpt_oauth_login_writes_auth_file(tmp_path, monkeypatch):
    import base64

    from free_claude_code.providers.chatgpt_oauth import oauth_login

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setattr(oauth_login, "CHATGPT_OAUTH_POLL_SAFETY_MS", 1)

    header = base64.urlsafe_b64encode(b'{}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_xyz"}}).encode()
    ).decode().rstrip("=")
    access_token = f"{header}.{payload}."

    responses = {
        ("POST", oauth_login.CHATGPT_OAUTH_DEVICE_URL): {
            "json": {
                "device_auth_id": "device_1",
                "user_code": "ABCD-EFGH",
                "interval": "1",
            }
        },
        ("POST", oauth_login.CHATGPT_OAUTH_DEVICE_TOKEN_URL): {
            "json": {
                "authorization_code": "auth_code_1",
                "code_verifier": "verifier_1",
            }
        },
        ("POST", oauth_login.CODEX_OAUTH_TOKEN_URL): {
            "json": {
                "access_token": access_token,
                "refresh_token": "refresh_1",
                "expires_in": 3600,
            }
        },
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def json(self):
            return self._payload["json"]

    class FakeClient:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._poll_count = 0

        def post(self, url, **kwargs):
            key = ("POST", url)
            self.calls.append(key)
            if url == oauth_login.CHATGPT_OAUTH_DEVICE_TOKEN_URL:
                self._poll_count += 1
            return FakeResponse(responses[key])

        def close(self):
            pass

    fake_client = FakeClient()
    tokens = oauth_login.perform_chatgpt_oauth_login(
        timeout_seconds=2,
        http_client=fake_client,  # type: ignore
    )

    assert tokens["access_token"] == access_token
    assert tokens["account_id"] == "acct_xyz"
    auth_file = tmp_path / ".codex" / "auth.json"
    assert auth_file.exists()
    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == access_token
    assert saved["tokens"]["refresh_token"] == "refresh_1"
    assert "expires_at" in saved["tokens"]


@pytest.mark.asyncio
async def test_stream_response_uses_send_not_stream_context_manager(chatgpt_oauth_provider):
    """Regression: awaiting httpx.AsyncClient.stream() raises TypeError."""
    from unittest.mock import MagicMock

    request = MessagesRequest(
        model="gpt-5",
        messages=[Message(role="user", content="hi")],
    )

    async def _raw_stream():
        yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
        yield b'data: {"type":"response.completed","response":{}}\n\n'

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.aiter_raw = _raw_stream
    fake_response.aclose = AsyncMock()

    client = chatgpt_oauth_provider._client
    client.build_request = MagicMock(return_value=MagicMock())
    client.send = AsyncMock(return_value=fake_response)

    chunks = [
        chunk
        async for chunk in chatgpt_oauth_provider.stream_response(
            request, request_id="req_1"
        )
    ]

    assert any("content_block_start" in chunk and "text" in chunk for chunk in chunks)
    assert any("text_delta" in chunk and "hello" in chunk for chunk in chunks)
    client.send.assert_awaited_once()
    assert client.send.await_args.kwargs.get("stream") is True
    fake_response.aclose.assert_awaited_once()
