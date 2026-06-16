from typing import cast
from unittest.mock import patch

import pytest
from fastapi.responses import StreamingResponse

from api.model_router import ResolvedModel
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.base import BaseProvider


class MockRepairProvider:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def cleanup(self):
        pass

    async def list_model_ids(self):
        return frozenset(["test"])

    def preflight_stream(self, *args, **kwargs):
        pass

    async def stream_response(self, request, **kwargs):
        resp = self.responses[self.call_count]
        self.call_count += 1
        for chunk in resp:
            yield chunk


@pytest.mark.asyncio
async def test_repair_malformed_json():
    settings = Settings()

    # First call returns malformed JSON in tool use
    # Second call (repair) returns fixed response
    resp1 = [
        'event: message_start\ndata: {"type": "message_start", "message": {"id": "m1", "type": "message", "role": "assistant", "content": [], "model": "test", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 1, "output_tokens": 1}}}\n\n',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "my_tool", "input": {}}}\n\n',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"arg\\": \\"unclosed"}} \n\n',  # Malformed JSON
        'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]
    resp2 = [
        'event: message_start\ndata: {"type": "message_start", "message": {"id": "m2", "role": "assistant"}}\n\n',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Repaired!"}}\n\n',
        'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]

    provider = MockRepairProvider([resp1, resp2])
    service = ClaudeProxyService(settings, lambda _: cast(BaseProvider, provider))

    with patch.object(service._model_router, "resolve_candidates") as mock_resolve:
        mock_resolve.return_value = [
            ResolvedModel(
                original_model="test",
                provider_id="test",
                provider_model="test",
                provider_model_ref="test/test",
                thinking_enabled=True,
            )
        ]

        req = MessagesRequest(
            model="repair/test", messages=[Message(role="user", content="test")]
        )
        res = await service.create_message(req)
        assert isinstance(res, StreamingResponse)

        # Since it's a repair/ prefix, create_message should return our RepairEngine's generator
        chunks = [str(chunk) for chunk in [chunk async for chunk in res.body_iterator]]

        full_output = "".join(chunks)
        assert "Repaired!" in full_output
        assert provider.call_count == 2
