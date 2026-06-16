from typing import cast

import pytest

from api.cache import ResponseCache
from api.models.anthropic import ContentBlockText, Message, MessagesRequest
from api.models.responses import MessagesResponse
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.base import BaseProvider


@pytest.mark.asyncio
async def test_non_streaming_cache():
    cache = ResponseCache(enabled=True)
    settings = Settings.model_construct(
        model="test",
        enable_response_caching=True,
    )

    # Mock data to cache
    request_data = MessagesRequest(
        model="test", messages=[Message(role="user", content="hello")], stream=False
    )

    cached_chunks = [
        'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "test"}}\n\n',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Cached response"}}\n\n',
        'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]

    cache.set(request_data, cached_chunks)

    service = ClaudeProxyService(
        settings, lambda _: cast(BaseProvider, None), cache=cache
    )

    response = await service.create_message(request_data)

    assert isinstance(response, MessagesResponse)
    assert response.id == "msg_123"
    block = response.content[0]
    assert isinstance(block, ContentBlockText)
    assert block.text == "Cached response"
