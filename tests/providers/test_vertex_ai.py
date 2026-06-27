"""Tests for the Vertex AI provider and its request/response handling."""

import pytest

from providers.vertex_ai.request import _openai_messages_to_contents


def test_openai_messages_to_contents_alternating_and_merging():
    # 1. Simple alternating roles
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    contents = _openai_messages_to_contents(messages)
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Hello"}]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": "Hi there!"}]
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [{"text": "How are you?"}]

    # 2. Consecutive user messages (should be merged)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Are you there?"},
    ]
    contents = _openai_messages_to_contents(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Hello"}, {"text": "Are you there?"}]

    # 3. Consecutive tool response messages (should be merged)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "list_dir",
                        "arguments": '{"DirectoryPath": "."}',
                    },
                },
                {
                    "id": "tc2",
                    "type": "function",
                    "function": {
                        "name": "view_file",
                        "arguments": '{"AbsolutePath": "CLAUDE.md"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "file1, file2",
        },
        {
            "role": "tool",
            "tool_call_id": "tc2",
            "content": "Line 1\nLine 2",
        },
    ]
    contents = _openai_messages_to_contents(messages)
    # The assistant message generates 1 model turn with tool calls
    # The two tool messages generate 1 user turn with two merged functionResponse parts
    assert len(contents) == 2
    assert contents[0]["role"] == "model"
    assert len(contents[0]["parts"]) == 2  # two functionCalls
    assert "functionCall" in contents[0]["parts"][0]
    assert "functionCall" in contents[0]["parts"][1]

    assert contents[1]["role"] == "user"
    assert len(contents[1]["parts"]) == 2  # two merged functionResponses
    assert contents[1]["parts"][0]["functionResponse"]["name"] == "list_dir"
    assert contents[1]["parts"][0]["functionResponse"]["response"] == {
        "content": "file1, file2"
    }
    assert contents[1]["parts"][1]["functionResponse"]["name"] == "view_file"
    assert contents[1]["parts"][1]["functionResponse"]["response"] == {
        "content": "Line 1\nLine 2"
    }

    # 4. Mixed user and tool message mapping safely (should not merge user text with tool functionResponse)
    messages = [
        {"role": "user", "content": "Text message"},
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "some tool output",
        },
    ]
    contents = _openai_messages_to_contents(messages)
    # Since one has functionResponse and the other has text, they should NOT merge into the same turn
    # even though both map to role 'user'. This avoids Gemini API 400 part mixing restrictions.
    assert len(contents) == 2
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Text message"}]
    assert contents[1]["role"] == "user"
    assert "functionResponse" in contents[1]["parts"][0]


def test_openai_messages_to_contents_strips_interrupted_and_no_content():
    # Test stripping of "[Tool use interrupted]" and "(no content)" from strings
    # Empty turns should be filtered out entirely, causing the remaining turns to be merged if they have the same role
    messages = [
        {"role": "user", "content": "[Tool use interrupted] drain context"},
        {"role": "assistant", "content": "(no content)"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello [Tool use interrupted]"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "world (no content)"}],
        },
    ]
    contents = _openai_messages_to_contents(messages)
    # The second message is assistant "(no content)" which becomes empty and is filtered.
    # The first and third messages are user and are consecutive after filtering, so they get merged.
    # The fourth message is assistant "world" which becomes model turn.
    assert len(contents) == 2

    # 1. Merge of 1 and 3: "drain context" and "hello"
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "drain context"}, {"text": "hello"}]

    # 2. Fourth message: "world"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": "world"}]


def test_vertex_ai_provider_missing_location_and_base_url():
    import pytest

    from providers.base import ProviderConfig
    from providers.exceptions import AuthenticationError
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_key",
        base_url="",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    with pytest.raises(AuthenticationError) as exc_info:
        VertexAIProvider(config, location="")
    assert "VERTEX_AI_BASE_URL or VERTEX_AI_LOCATION must be set" in str(exc_info.value)


@pytest.mark.anyio
async def test_vertex_ai_provider_stream_response_error_propagation() -> None:
    from unittest.mock import AsyncMock, patch

    import httpx

    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_key",
        base_url="https://aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    provider = VertexAIProvider(config, location="us-central1")

    # Mock send to return a 404 response
    mock_send = AsyncMock()
    mock_response = httpx.Response(
        status_code=404,
        request=httpx.Request("POST", "http://test"),
        content=b'{"error": {"message": "Model gemini-3.5-flash not found", "status": "NOT_FOUND"}}',
    )
    mock_send.return_value = mock_response

    class DummyRequest:
        model = "vertex_ai/google/gemini-3.5-flash"
        system = ""
        max_tokens = 100
        stream = True
        extra_headers = None
        extra_query = None
        extra_body = None
        stop = None
        temperature = None
        top_p = None
        top_k = None
        tools = None
        tool_choice = None

        def __init__(self) -> None:
            self.messages: list = []

    with patch.object(provider._client, "send", mock_send):
        events = [event async for event in provider.stream_response(DummyRequest())]

    # Verify that the mapped error message containing model-not-found detail was yielded
    combined_events = "".join(events)
    assert "Model gemini-3.5-flash not found" in combined_events
    assert "Upstream provider VERTEX_AI returned HTTP 404." in combined_events

    await provider.cleanup()


@pytest.mark.anyio
async def test_vertex_ai_provider_request_parameters_and_headers() -> None:
    from unittest.mock import AsyncMock, patch

    import httpx

    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    provider = VertexAIProvider(config, location="us-central1")

    # Mock send to check the built request
    mock_send = AsyncMock()
    mock_send.return_value = httpx.Response(200, json={})

    class DummyRequest:
        model = "vertex_ai/google/gemini-3.5-flash"
        system = ""
        max_tokens = 100
        stream = True
        extra_headers = None
        extra_query = None
        extra_body = None
        stop = None
        temperature = None
        top_p = None
        top_k = None
        tools = None
        tool_choice = None

        def __init__(self) -> None:
            self.messages: list = []

    # Patch build_request to inspect it or patch send to inspect it
    with patch.object(provider._client, "send", mock_send):
        try:
            async for _ in provider.stream_response(DummyRequest()):
                pass
        except Exception:
            pass

    # Inspect the request that was built and sent
    assert mock_send.called
    called_request = mock_send.call_args[0][0]

    # Check url query parameters: should contain alt=sse but NOT contain key
    query_params = dict(called_request.url.params)
    assert query_params.get("alt") == "sse"
    assert "key" not in query_params

    # Check headers: should contain x-goog-api-key
    assert called_request.headers.get("x-goog-api-key") == "test_api_key_12345"

    await provider.cleanup()


@pytest.mark.anyio
async def test_vertex_ai_provider_request_headers_with_oauth_token() -> None:
    from unittest.mock import AsyncMock, patch

    import httpx

    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="ya29.test_oauth_token_12345",
        base_url="https://aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    provider = VertexAIProvider(config, location="us-central1")

    mock_send = AsyncMock()
    mock_send.return_value = httpx.Response(200, json={})

    class DummyRequest:
        model = "vertex_ai/google/gemini-3.5-flash"
        system = ""
        max_tokens = 100
        stream = True
        extra_headers = None
        extra_query = None
        extra_body = None
        stop = None
        temperature = None
        top_p = None
        top_k = None
        tools = None
        tool_choice = None

        def __init__(self) -> None:
            self.messages: list = []

    with patch.object(provider._client, "send", mock_send):
        try:
            async for _ in provider.stream_response(DummyRequest()):
                pass
        except Exception:
            pass

    assert mock_send.called
    called_request = mock_send.call_args[0][0]

    # Check headers: should contain Authorization and NOT contain x-goog-api-key
    assert (
        called_request.headers.get("Authorization")
        == "Bearer ya29.test_oauth_token_12345"
    )
    assert "x-goog-api-key" not in called_request.headers

    await provider.cleanup()


@pytest.mark.anyio
async def test_vertex_ai_provider_project_id_path() -> None:
    from unittest.mock import AsyncMock, patch

    import httpx

    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    provider = VertexAIProvider(
        config, project_id="test_project", location="us-central1"
    )

    # Mock send to check the built request
    mock_send = AsyncMock()
    mock_send.return_value = httpx.Response(200, json={})

    class DummyRequest:
        model = "vertex_ai/google/gemini-3.5-flash"
        system = ""
        max_tokens = 100
        stream = True
        extra_headers = None
        extra_query = None
        extra_body = None
        stop = None
        temperature = None
        top_p = None
        top_k = None
        tools = None
        tool_choice = None

        def __init__(self) -> None:
            self.messages: list = []

    with patch.object(provider._client, "send", mock_send):
        try:
            async for _ in provider.stream_response(DummyRequest()):
                pass
        except Exception:
            pass

    assert mock_send.called
    called_request = mock_send.call_args[0][0]

    # Check url path: should contain project_id and location segment
    # Note that base_url has /v1 at the end. Since the request path does not begin with /v1, we need to verify how it resolves.
    assert (
        called_request.url.path
        == "/v1/projects/test_project/locations/us-central1/publishers/google/models/gemini-3.5-flash:streamGenerateContent"
    )

    await provider.cleanup()


@pytest.mark.anyio
async def test_vertex_ai_provider_list_models() -> None:
    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    provider = VertexAIProvider(config, location="us-central1")

    model_ids = await provider.list_model_ids()
    assert "google/gemini-3.5-flash" in model_ids
    assert "google/gemini-2.5-pro" in model_ids

    # Verify non-chat, embedding, and video models are NOT in the list
    assert "google/gemini-embedding-001" not in model_ids
    assert "google/text-embedding-005" not in model_ids
    assert "google/veo-3.1-generate-001" not in model_ids
    assert "google/gemini-live-2.5-flash-native-audio" not in model_ids
    assert "google/gemini-2.5-flash-image" not in model_ids

    model_infos = await provider.list_model_infos()
    assert len(model_infos) == len(model_ids)
    info_ids = {info.model_id for info in model_infos}
    assert info_ids == model_ids

    await provider.cleanup()


def test_vertex_ai_provider_rejects_openapi_base_url() -> None:
    from providers.base import ProviderConfig
    from providers.exceptions import AuthenticationError
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/endpoints/openapi",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    with pytest.raises(AuthenticationError) as exc_info:
        VertexAIProvider(config, location="us-central1")
    assert "cannot be an OpenAI-compatible endpoint" in str(exc_info.value)


@pytest.mark.anyio
async def test_vertex_ai_raw_stream_logging_gate(caplog) -> None:
    import logging
    from unittest.mock import AsyncMock, patch

    import httpx

    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    # Case 1: log_raw_sse_events=False
    config = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
        log_raw_sse_events=False,
    )
    provider = VertexAIProvider(config, location="us-central1")
    mock_send = AsyncMock()
    mock_send.return_value = httpx.Response(
        200,
        content=b'data: {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}\n\n',
    )

    class DummyRequest:
        model = "vertex_ai/google/gemini-3.5-flash"
        system = ""
        max_tokens = 100
        stream = True
        extra_headers = None
        extra_query = None
        extra_body = None
        stop = None
        temperature = None
        top_p = None
        top_k = None
        tools = None
        tool_choice = None

        def __init__(self) -> None:
            self.messages: list = []

    with (
        patch.object(provider._client, "send", mock_send),
        caplog.at_level(logging.DEBUG),
    ):
        async for _ in provider.stream_response(DummyRequest()):
            pass

    # Should NOT contain VERTEX_AI_RAW_STREAM_DATA
    assert "VERTEX_AI_RAW_STREAM_DATA" not in caplog.text

    # Case 2: log_raw_sse_events=True
    config_enabled = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
        log_raw_sse_events=True,
    )
    provider_enabled = VertexAIProvider(config_enabled, location="us-central1")

    with (
        patch.object(provider_enabled._client, "send", mock_send),
        caplog.at_level(logging.DEBUG),
    ):
        caplog.clear()
        async for _ in provider_enabled.stream_response(DummyRequest()):
            pass

    # Should contain VERTEX_AI_RAW_STREAM_DATA
    assert "VERTEX_AI_RAW_STREAM_DATA" in caplog.text

    await provider.cleanup()
    await provider_enabled.cleanup()


@pytest.mark.anyio
async def test_vertex_ai_provider_request_sent_trace_body_snapshot() -> None:
    from unittest.mock import AsyncMock, patch

    import httpx

    from providers.base import ProviderConfig
    from providers.vertex_ai import VertexAIProvider

    config = ProviderConfig(
        api_key="test_api_key_12345",
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
        rate_limit=1,
        rate_window=1,
        max_concurrency=1,
        http_read_timeout=10,
        http_write_timeout=10,
        http_connect_timeout=10,
        enable_thinking=False,
    )
    provider = VertexAIProvider(config, location="us-central1")
    mock_send = AsyncMock()
    mock_send.return_value = httpx.Response(200, json={})

    class DummyRequest:
        model = "vertex_ai/google/gemini-3.5-flash"
        system = ""
        max_tokens = 100
        stream = True
        extra_headers = None
        extra_query = None
        extra_body = None
        stop = None
        temperature = None
        top_p = None
        top_k = None
        tools = None
        tool_choice = None

        def __init__(self) -> None:
            self.messages: list = []

    with (
        patch("providers.vertex_ai.client.trace_event") as mock_trace_event,
        patch.object(provider._client, "send", mock_send),
    ):
        try:
            async for _ in provider.stream_response(DummyRequest()):
                pass
        except Exception:
            pass

    # Check trace_event calls
    assert mock_trace_event.called
    # Find the call for provider.request.sent
    sent_call = None
    for call in mock_trace_event.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("event") == "provider.request.sent":
            sent_call = kwargs
            break

    assert sent_call is not None
    body_snap = sent_call["body"]
    # Check that it is a snapshot containing only allowed keys
    assert "contents" in body_snap
    assert "generationConfig" in body_snap

    await provider.cleanup()


def test_build_request_body_system_instruction():
    from providers.vertex_ai.request import build_request_body

    class MockMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class MockRequest:
        def __init__(self):
            self.model = "vertex_ai/google/gemini-3.5-flash"
            self.messages = [MockMessage("user", "Hello")]
            self.system = "Be helpful."
            self.max_tokens = 100
            self.stop = None
            self.temperature = None
            self.top_p = None
            self.top_k = None
            self.tools = None
            self.tool_choice = None

    request = MockRequest()
    body = build_request_body(request, thinking_enabled=False)

    assert "systemInstruction" in body
    assert body["systemInstruction"] == {"parts": [{"text": "Be helpful."}]}
    assert "system_instruction" not in body
    assert "generationConfig" in body
    assert "thinkingConfig" not in body["generationConfig"]

    body_with_think = build_request_body(request, thinking_enabled=True)
    assert "generationConfig" in body_with_think
    assert "thinkingConfig" in body_with_think["generationConfig"]
    assert body_with_think["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 2048
    }


def test_save_thought_signature_concurrent(tmp_path):
    import concurrent.futures
    from unittest.mock import patch

    from providers.vertex_ai.request import save_thought_signature

    temp_cache_file = tmp_path / "vertex_ai_signatures.json"

    with patch(
        "providers.vertex_ai.request._get_cache_file_path", return_value=temp_cache_file
    ):

        def save_one(i):
            save_thought_signature(f"tool_{i}", {"arg": i}, f"sig_{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(save_one, i) for i in range(50)]
            concurrent.futures.wait(futures)

        for fut in futures:
            fut.result()

        assert temp_cache_file.exists()
        import json

        with open(temp_cache_file, encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            for i in range(50):
                expected_key = f'tool_{i}:{{"arg": {i}}}'
                assert data.get(expected_key) == f"sig_{i}"


@pytest.mark.anyio
async def test_flush_thought_signatures(tmp_path) -> None:
    from unittest.mock import patch

    from providers.vertex_ai.request import (
        flush_thought_signatures,
        save_thought_signature,
    )

    temp_cache_file = tmp_path / "vertex_ai_signatures.json"

    with patch(
        "providers.vertex_ai.request._get_cache_file_path", return_value=temp_cache_file
    ):
        save_thought_signature("test_tool", {"param": 1}, "test_sig_value")
        await flush_thought_signatures()

        assert temp_cache_file.exists()
        import json

        with open(temp_cache_file, encoding="utf-8") as f:
            data = json.load(f)
            assert data.get('test_tool:{"param": 1}') == "test_sig_value"
