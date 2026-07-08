"""Tests for DeepSeek OpenAI-compatible Chat Completions provider."""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.models.anthropic import (
    ContentBlockImage,
    Message,
    MessagesRequest,
    Tool,
)
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.stream_contracts import parse_sse_text
from providers.base import ProviderConfig
from providers.deepseek import (
    DEEPSEEK_DEFAULT_BASE,
    DeepSeekProvider,
)
from providers.exceptions import InvalidRequestError


@pytest.fixture
def deepseek_config():
    return ProviderConfig(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config)


def test_default_base_url_alias():
    assert DEEPSEEK_DEFAULT_BASE == "https://api.deepseek.com"


def test_init(deepseek_config):
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_client:
        provider = DeepSeekProvider(deepseek_config)
    assert provider._api_key == "test_deepseek_key"
    assert provider._base_url == "https://api.deepseek.com"
    assert mock_client.called


def test_build_request_body_openai_chat_shape(deepseek_provider):
    request = MessagesRequest(
        model="deepseek-v4-pro",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="S",
    )
    body = deepseek_provider._build_request_body(request)
    assert body["model"] == "deepseek-v4-pro"
    assert "stream" not in body
    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100
    assert "stream_options" not in body


def test_build_request_body_default_max_tokens(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = deepseek_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["extra_body"]["thinking"] == {"type": "enabled"}


def test_build_request_body_tool_list_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["tools"][0]["function"]["name"] == "Read"


def test_build_request_body_tool_choice_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["tool_choice"] == "auto"


def test_build_request_body_forced_tool_choice_downgrades_to_auto(
    deepseek_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "tool", "name": "Read"},
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["tool_choice"] == "auto"


def test_build_request_body_respects_global_thinking_disable():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._build_request_body(request)
    assert "extra_body" not in body
    assert "stream_options" not in body


def test_preserve_unsigned_thinking_when_thinking_on(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "plain",
                            "signature": None,
                        },
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["messages"][0]["content"] == "out"
    assert body["messages"][0]["reasoning_content"] == "plain"


def test_strip_redacted_thinking_when_thinking_on(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["messages"][0] == {"role": "assistant", "content": "out"}


def test_tool_history_with_replayable_thinking_preserves_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "hidden",
                            "signature": "sig_123",
                        },
                        {"type": "redacted_thinking", "data": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "output_config": {"effort": "high"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert "context_management" not in body
    assert "output_config" not in body
    assistant = body["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "hidden"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"file_path": "x"}'
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "ok",
    }


def test_tool_history_with_unsigned_thinking_preserves_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plain"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["reasoning_content"] == "plain"


def test_tool_history_without_thinking_disables_thinking_and_hints(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "other_edit", "keep": "all"},
                ],
                "other": True,
            },
            "output_config": {"effort": "high", "format": "text"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert "extra_body" not in body
    assert "context_management" not in body
    assert "output_config" not in body
    assert body["tools"][0]["function"]["name"] == "Read"
    assert body["tool_choice"] == "auto"
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"
    assert body["messages"][1]["role"] == "tool"


def test_tool_history_with_empty_thinking_preserves_reasoning_state(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["reasoning_content"] == ""
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"


def test_tool_history_with_empty_top_level_reasoning_preserves_reasoning_state(
    deepseek_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                    "reasoning_content": "",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["reasoning_content"] == ""
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"


def test_thinking_off_strips_thinking_history():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "sec"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
        }
    )
    body = provider._build_request_body(request)
    assert "reasoning_content" not in body["messages"][0]
    assert "sec" not in str(body["messages"])


def test_passthrough_tool_use_and_result(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "n",
                            "input": {"a": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "n"
    assert body["messages"][1]["role"] == "tool"


def test_preflight_strips_user_image():
    """Image blocks are silently stripped (DeepSeek lacks vision); request must not fail."""
    request = MessagesRequest(
        model="m",
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "YQ==",
                        },
                    )
                ],
            )
        ],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    # Should not raise; image is stripped.
    provider.preflight_stream(request, thinking_enabled=True)
    body = provider._build_request_body(request)
    content = body["messages"][0]["content"]
    assert "attachment omitted" in content.lower()
    assert "image or document inputs" in content.lower()


def test_preflight_rejects_mcp_servers():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        mcp_servers=[{"type": "url", "url": "https://x"}],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="mcp_servers"):
        provider.preflight_stream(request)


def test_preflight_rejects_listed_server_tools_in_tools_list():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        tools=[Tool(name="web_search", type="web_search_20250305", input_schema={})],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="web_search"):
        provider.preflight_stream(request)


def test_preflight_rejects_server_tool_result_blocks():
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"q": "a"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s1",
                            "content": [],
                        },
                    ],
                }
            ],
        }
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match=r"web_search_tool_result|server"):
        provider.preflight_stream(request)


def test_reasoning_content_replayed_to_openai_chat(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[
            Message(
                role="assistant",
                content="hi",
                reasoning_content="r",
            )
        ],
    )
    body = deepseek_provider._build_request_body(request)
    assert body["messages"][0]["reasoning_content"] == "r"


@pytest.mark.asyncio
async def test_stream_uses_chat_completions_and_maps_cache_usage(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )

    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="hello", reasoning_content=None, tool_calls=None
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None, reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                completion_tokens=3,
                prompt_tokens=30,
                prompt_cache_hit_tokens=10,
                prompt_cache_miss_tokens=20,
            ),
        )

    create = AsyncMock(return_value=fake_stream())
    with patch.object(deepseek_provider._client.chat.completions, "create", create):
        chunks = [
            chunk
            async for chunk in deepseek_provider.stream_response(
                request, input_tokens=7, request_id="r1"
            )
        ]

    create.assert_awaited_once()
    await_args = create.await_args
    assert await_args is not None
    assert await_args.kwargs["model"] == "m"
    assert await_args.kwargs["stream"] is True
    assert await_args.kwargs["stream_options"] == {"include_usage": True}
    parsed = parse_sse_text("".join(chunks))
    usage = next(
        event.data["usage"] for event in parsed if event.event == "message_delta"
    )
    assert usage == {
        "input_tokens": 30,
        "output_tokens": 3,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 20,
    }


def test_preserves_extra_body_for_openai_chat_request(deepseek_provider):
    raw = {
        "model": "m",
        "max_tokens": 3,
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"note": 1},
    }
    r = MessagesRequest.model_validate(raw)
    body = deepseek_provider._build_request_body(r)
    assert body["extra_body"] == {"note": 1, "thinking": {"type": "enabled"}}


def test_normalizes_tool_result_content_array_to_string(deepseek_provider):
    """Test that tool_result content arrays are normalized to strings for DeepSeek API."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "list_dir",
                            "input": {"path": "/"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "file1.txt"},
                                {"type": "text", "text": "file2.txt"},
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert "file1.txt" in tool_result["content"]
    assert "file2.txt" in tool_result["content"]


def test_strips_document_blocks_for_deepseek(deepseek_provider):
    """Document blocks (e.g. PDFs from Claude Code) are stripped since DeepSeek can't process them."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "PDF text extracted",
                        },
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_abc"},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["messages"][0] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "PDF text extracted",
    }


def test_strips_image_blocks_for_deepseek(deepseek_provider):
    """Image blocks are stripped for DeepSeek since it doesn't support vision."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["messages"][0] == {"role": "user", "content": "describe this"}


def test_normalizes_tool_result_content_dict_to_string(deepseek_provider):
    """Test that tool_result content dicts are normalized to JSON strings."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_data",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": {"status": "success", "data": [1, 2, 3]},
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert "status" in tool_result["content"]
    assert "success" in tool_result["content"]


def test_strips_image_block_inside_tool_result(deepseek_provider):
    """Image blocks nested inside tool_result.content are stripped, not rejected."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"path": "shot.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "screenshot saved"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    # After stripping + string-normalization, no base64/image marker survives.
    assert isinstance(tool_result["content"], str)
    assert "screenshot saved" in tool_result["content"]
    assert "base64" not in tool_result["content"]
    assert "abc" not in tool_result["content"]


def test_image_only_tool_result_replaced_with_placeholder(deepseek_provider):
    """A tool_result whose only inner block is an image becomes a placeholder string."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Screenshot",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert tool_result["content"] != ""
    assert "attachment omitted" in tool_result["content"].lower()
    assert "image or document inputs" in tool_result["content"].lower()


def test_document_only_tool_result_replaced_with_generic_placeholder(
    deepseek_provider,
):
    """A document-only tool_result uses the generic attachment placeholder."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "paper.pdf"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "file",
                                        "file_id": "file_pdf",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert "attachment omitted" in tool_result["content"].lower()
    assert "document inputs" in tool_result["content"].lower()
    assert "image omitted" not in tool_result["content"].lower()


def test_image_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level image-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][0]["content"]
    assert "attachment omitted" in content.lower()
    assert "image or document inputs" in content.lower()


def test_document_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level document-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_pdf"},
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    content = body["messages"][0]["content"]
    assert "attachment omitted" in content.lower()
    assert "document inputs" in content.lower()


def test_warns_when_stripping_attachment_blocks(deepseek_provider, caplog):
    """A warning is emitted when image/document blocks are dropped so users notice."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Screenshot",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stripped unsupported attachment blocks" in r.message for r in warnings)


def test_no_warning_when_no_attachments(deepseek_provider, caplog):
    """No warning is emitted on plain text-only requests."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)

    assert not any(
        "stripped unsupported attachment blocks" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
