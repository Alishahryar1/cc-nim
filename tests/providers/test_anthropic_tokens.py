from unittest.mock import MagicMock, patch

import httpx
import pytest

from free_claude_code.core.anthropic.models import Message, SystemContent, Tool
from free_claude_code.providers.anthropic_tokens import (
    AnthropicTokenCountUnavailable,
    count_tokens_via_anthropic_api,
)


def _response(input_tokens: object = 42) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"input_tokens": input_tokens}
    return response


def _call(**overrides: object) -> int:
    kwargs = {
        "api_key": "sk-test",
        "model": "claude-sonnet-4-5-20250929",
        "messages": [Message(role="user", content="hi")],
        "system": None,
        "tools": None,
        "timeout": 30.0,
        "proxy": "",
    }
    kwargs.update(overrides)
    return count_tokens_via_anthropic_api(**kwargs)  # type: ignore[arg-type]


def test_returns_exact_count_and_builds_expected_request() -> None:
    response = _response(17)
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response
    with patch(
        "free_claude_code.providers.anthropic_tokens.httpx.Client",
        return_value=client,
    ) as client_cls:
        assert _call() == 17

    client_cls.assert_called_once_with(proxy=None, timeout=30.0)
    response.raise_for_status.assert_called_once()
    request = client.post.call_args
    assert request.args[0].endswith("/v1/messages/count_tokens")
    assert request.kwargs["headers"]["x-api-key"] == "sk-test"
    assert request.kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_passes_proxy_to_http_client() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = _response()
    with patch(
        "free_claude_code.providers.anthropic_tokens.httpx.Client",
        return_value=client,
    ) as client_cls:
        _call(proxy="socks5://127.0.0.1:1080")
    client_cls.assert_called_once_with(
        proxy="socks5://127.0.0.1:1080",
        timeout=30.0,
    )


def test_serializes_optional_system_and_tools() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = _response()
    with patch(
        "free_claude_code.providers.anthropic_tokens.httpx.Client",
        return_value=client,
    ):
        _call(
            system=[SystemContent(type="text", text="Be concise.")],
            tools=[Tool(name="lookup", input_schema={"type": "object"})],
        )
    payload = client.post.call_args.kwargs["json"]
    assert payload["system"] == [{"type": "text", "text": "Be concise."}]
    assert payload["tools"] == [
        {"name": "lookup", "input_schema": {"type": "object"}}
    ]


def test_omits_absent_optional_fields() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = _response()
    with patch(
        "free_claude_code.providers.anthropic_tokens.httpx.Client",
        return_value=client,
    ):
        _call()
    payload = client.post.call_args.kwargs["json"]
    assert "system" not in payload
    assert "tools" not in payload


def test_wraps_network_and_http_failures() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com")
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = httpx.ConnectError("offline", request=request)
    with (
        patch(
            "free_claude_code.providers.anthropic_tokens.httpx.Client",
            return_value=client,
        ),
        pytest.raises(AnthropicTokenCountUnavailable),
    ):
        _call()


@pytest.mark.parametrize("value", [None, "42", True])
def test_rejects_invalid_input_token_response(value: object) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = _response(value)
    with (
        patch(
            "free_claude_code.providers.anthropic_tokens.httpx.Client",
            return_value=client,
        ),
        pytest.raises(AnthropicTokenCountUnavailable),
    ):
        _call()


def test_payload_programming_error_is_not_hidden_by_fallback() -> None:
    message = MagicMock()
    message.model_dump.side_effect = RuntimeError("serialization bug")
    with pytest.raises(RuntimeError, match="serialization bug"):
        _call(messages=[message])
