import pytest
from fastapi.responses import StreamingResponse

from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.base import BaseProvider


class MockStreamProvider(BaseProvider):
    def __init__(self, config, text):
        super().__init__(config)
        self.text = text

    async def cleanup(self):
        pass

    async def list_model_ids(self):
        return frozenset(["m"])

    async def stream_response(
        self,
        request,
        input_tokens=0,
        *,
        request_id=None,
        thinking_enabled=None,
    ):
        # Very simple mock SSE
        yield f'data: {{"type": "content_block_delta", "delta": {{"type": "text", "text": "{self.text}"}}}}\n\n'


@pytest.mark.asyncio
async def test_fusion_mode():
    settings = Settings.model_construct(
        model="open_router/m1", enable_response_caching=False
    )

    # Setup mock providers for panel and judge
    provider_panel1 = MockStreamProvider(None, "Panel 1 Response")
    provider_panel2 = MockStreamProvider(None, "Panel 2 Response")
    provider_judge = MockStreamProvider(None, "Synthesized Result")

    def provider_getter(provider_id):
        if provider_id == "mistral":
            return provider_panel1
        if provider_id == "deepseek":
            return provider_panel2
        return provider_judge

    service = ClaudeProxyService(settings, provider_getter)

    # Trigger fusion via model name
    # Format: fusion/judge_model:panel_model1,panel_model2
    request_data = MessagesRequest(
        model="fusion/open_router/j1:mistral/m1,deepseek/m2",
        messages=[Message(role="user", content="Tell me a joke")],
    )

    response = await service.create_message(request_data)
    assert isinstance(response, StreamingResponse)

    full_text = "".join([str(chunk) async for chunk in response.body_iterator])

    assert "Synthesized Result" in full_text


@pytest.mark.asyncio
async def test_named_fusion_panel():
    settings = Settings.model_construct(
        model="open_router/fallback",
        enable_response_caching=False,
        fusion_panels="coding=open_router/j1:mistral/m1,deepseek/m2",
    )

    provider_judge = MockStreamProvider(None, "Named Panel Success")

    def provider_getter(provider_id):
        if provider_id == "mistral":
            return MockStreamProvider(None, "m1")
        if provider_id == "deepseek":
            return MockStreamProvider(None, "m2")
        return provider_judge

    service = ClaudeProxyService(settings, provider_getter)

    # Trigger fusion via named panel
    request_data = MessagesRequest(
        model="fusion/coding",
        messages=[Message(role="user", content="Refactor this")],
    )

    response = await service.create_message(request_data)
    assert isinstance(response, StreamingResponse)

    full_text = "".join([str(chunk) async for chunk in response.body_iterator])

    assert "Named Panel Success" in full_text
