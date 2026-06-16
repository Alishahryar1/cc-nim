import pytest
from fastapi.responses import StreamingResponse

from api.model_router import ModelRouter
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.base import BaseProvider
from providers.exceptions import ProviderError


class MockProvider(BaseProvider):
    def __init__(self, config, status_code=200):
        super().__init__(config)
        self.status_code = status_code
        self.call_count = 0

    async def cleanup(self):
        pass

    async def list_model_ids(self):
        return frozenset(["model1"])

    def preflight_stream(self, *args, **kwargs):
        pass

    async def stream_response(
        self, request, input_tokens=0, request_id=None, thinking_enabled=None
    ):
        self.call_count += 1
        if self.status_code != 200:
            raise ProviderError("Error", status_code=self.status_code)
        yield 'data: {"type": "message_start"}\n\n'


@pytest.mark.asyncio
async def test_automatic_failover():
    # Setup settings with failover
    settings = Settings.model_construct(
        model="provider1/model1,provider2/model2",
        enable_web_server_tools=False,
        log_api_error_tracebacks=True,
        log_raw_api_payloads=False,
        enable_model_thinking=True,
        enable_response_caching=False,
    )

    router = ModelRouter(settings)

    # Mock providers: first fails, second succeeds
    provider1 = MockProvider(None, status_code=503)
    provider2 = MockProvider(None, status_code=200)

    def provider_getter(provider_id):
        if provider_id == "provider1":
            return provider1
        return provider2

    service = ClaudeProxyService(settings, provider_getter, model_router=router)

    request_data = MessagesRequest(
        model="claude-3-sonnet-20240229",
        messages=[Message(role="user", content="hello")],
    )

    response = await service.create_message(request_data)
    assert isinstance(response, StreamingResponse)

    # Consume the streaming response to trigger the calls
    chunks = [chunk async for chunk in response.body_iterator]

    assert provider1.call_count == 1
    assert provider2.call_count == 1
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_failover_exhausted():
    # Setup settings with failover
    settings = Settings.model_construct(
        model="provider1/model1,provider2/model2",
        enable_web_server_tools=False,
        log_api_error_tracebacks=True,
        log_raw_api_payloads=False,
        enable_model_thinking=True,
        enable_response_caching=False,
    )

    router = ModelRouter(settings)

    # Mock providers: both fail
    provider1 = MockProvider(None, status_code=503)
    provider2 = MockProvider(None, status_code=503)

    def provider_getter(provider_id):
        if provider_id == "provider1":
            return provider1
        return provider2

    service = ClaudeProxyService(settings, provider_getter, model_router=router)

    request_data = MessagesRequest(
        model="claude-3-sonnet-20240229",
        messages=[Message(role="user", content="hello")],
    )

    # For streaming responses, ProviderError is raised during stream priming
    with pytest.raises(ProviderError) as excinfo:
        await service.create_message(request_data)

    assert excinfo.value.status_code == 503
    assert provider1.call_count == 1
    assert provider2.call_count == 1
