"""Tests for the dynamic vision fallback routing logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.models.anthropic import (
    ContentBlockImage,
    ContentBlockText,
    Message,
    MessagesRequest,
)
from api.services import ClaudeProxyService
from config.settings import Settings


def test_vision_fallback_skips_when_no_image():
    settings = Settings()
    settings.model = "open_router/openai/gpt-oss-120b:free"
    settings.model_fallback_vision = "open_router/moonshotai/kimi-k2.6:free"

    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: message_start\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)

    request = MessagesRequest(
        model="claude-3-sonnet-20240229",
        max_tokens=10,
        messages=[Message(role="user", content="hello text only")],
    )

    with patch("api.services.logger.info") as mock_info:
        service.create_message(request)

    # Check that fallback was NOT logged
    fallback_logs = [
        call.args
        for call in mock_info.call_args_list
        if "Dynamically falling back to vision model" in str(call.args)
    ]
    assert len(fallback_logs) == 0


def test_vision_fallback_triggers_when_image_present():
    settings = Settings()
    settings.model = "open_router/openai/gpt-oss-120b:free"
    settings.model_fallback_vision = "open_router/moonshotai/kimi-k2.6:free"

    mock_provider = MagicMock()
    # provider_getter will receive the provider_id ("open_router")
    provider_ids_requested = []

    def get_provider(pid):
        provider_ids_requested.append(pid)
        return mock_provider

    async def fake_stream(*_a, **_kw):
        yield "event: message_start\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=get_provider)

    # Message containing an image block as content dict
    request = MessagesRequest(
        model="claude-3-sonnet-20240229",
        max_tokens=10,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="What is this image?"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    ),
                ],
            )
        ],
    )

    with patch("api.services.logger.info") as mock_info:
        service.create_message(request)

    # Check that fallback WAS logged
    fallback_logs = [
        call.args
        for call in mock_info.call_args_list
        if "Dynamically falling back to vision model" in str(call.args[0])
    ]
    assert len(fallback_logs) == 1
    # Verify the fallback model is logged correctly
    assert "open_router/moonshotai/kimi-k2.6:free" in fallback_logs[0][2]


def test_vision_fallback_ignored_when_model_already_supports_vision():
    settings = Settings()
    settings.model = "open_router/google/gemini-2.5-flash:free"
    settings.model_fallback_vision = "open_router/moonshotai/kimi-k2.6:free"

    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: message_start\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)

    request = MessagesRequest(
        model="claude-3-sonnet-20240229",
        max_tokens=10,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="What is this image?"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    ),
                ],
            )
        ],
    )

    with patch("api.services.logger.info") as mock_info:
        service.create_message(request)

    # Check that fallback was NOT logged
    fallback_logs = [
        call.args
        for call in mock_info.call_args_list
        if "Dynamically falling back to vision model" in str(call.args)
    ]
    assert len(fallback_logs) == 0


def test_vision_fallback_ignored_when_no_fallback_configured():
    settings = Settings()
    settings.model = "open_router/openai/gpt-oss-120b:free"
    settings.model_fallback_vision = None

    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: message_start\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)

    request = MessagesRequest(
        model="claude-3-sonnet-20240229",
        max_tokens=10,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="What is this image?"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    ),
                ],
            )
        ],
    )

    with patch("api.services.logger.info") as mock_info:
        service.create_message(request)

    # Check that fallback was NOT logged
    fallback_logs = [
        call.args
        for call in mock_info.call_args_list
        if "Dynamically falling back to vision model" in str(call.args)
    ]
    assert len(fallback_logs) == 0


def test_vision_fallback_sanitizes_multiple_images():
    settings = Settings()
    settings.model = "open_router/openai/gpt-oss-120b:free"
    settings.model_fallback_vision = "open_router/moonshotai/kimi-k2.6:free"

    mock_provider = MagicMock()
    captured_body = {}

    async def fake_stream(request, *args, **kwargs):
        captured_body["request"] = request
        yield "event: message_start\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)

    request = MessagesRequest(
        model="claude-3-sonnet-20240229",
        max_tokens=10,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="First turn image"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "first_img_data",
                        },
                    ),
                ],
            ),
            Message(
                role="assistant",
                content="Understood.",
            ),
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="Second turn image"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "second_img_data",
                        },
                    ),
                ],
            ),
        ],
    )

    response = service.create_message(request)

    import asyncio
    from fastapi.responses import StreamingResponse

    async def consume():
        if isinstance(response, StreamingResponse):
            async for _ in response.body_iterator:
                pass

    asyncio.run(consume())

    routed_req = captured_body.get("request")
    assert routed_req is not None
    # Verify the first image block was replaced by a text block placeholder
    first_msg = routed_req.messages[0]
    assert first_msg.content[1].type == "text"
    assert "removed to comply with 1-image limit" in first_msg.content[1].text

    # Verify the second (most recent) image block was kept as-is
    last_msg = routed_req.messages[2]
    assert last_msg.content[1].type == "image"
    assert last_msg.content[1].source["data"] == "second_img_data"
