import pytest
import shutil
import os
from api.services import ClaudeProxyService
from api.model_router import ModelRouter
from api.models.anthropic import MessagesRequest
from config.settings import Settings
from api.cache import ResponseCache
from providers.base import BaseProvider

class MockProvider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.call_count = 0

    async def cleanup(self):
        pass

    async def list_model_ids(self):
        return frozenset(["model1"])

    async def stream_response(self, request, input_tokens=0, request_id=None, thinking_enabled=None):
        self.call_count += 1
        yield f"data: {{\"type\": \"content_block_delta\", \"delta\": {{\"type\": \"text\", \"text\": \"Response {self.call_count}\"}}}}\n\n"

@pytest.mark.asyncio
async def test_caching():
    cache_dir = ".test_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    cache = ResponseCache(cache_dir=cache_dir, enabled=True)
    settings = Settings.model_construct(
        model="provider1/model1",
        enable_web_server_tools=False,
        log_api_error_tracebacks=True,
        log_raw_api_payloads=False,
        enable_model_thinking=True,
        enable_response_caching=True
    )

    provider = MockProvider(None)
    def provider_getter(provider_id):
        return provider

    service = ClaudeProxyService(settings, provider_getter, cache=cache)

    request_data = MessagesRequest(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "hello"}]
    )

    # First call - should go to provider
    response1 = await service.create_message(request_data)
    chunks1 = []
    async for chunk in response1.body_iterator:
        chunks1.append(chunk)

    assert provider.call_count == 1
    assert "Response 1" in chunks1[0]

    # Second call - should come from cache
    response2 = await service.create_message(request_data)
    chunks2 = []
    async for chunk in response2.body_iterator:
        chunks2.append(chunk)

    assert provider.call_count == 1  # Still 1
    assert "Response 1" in chunks2[0]

    shutil.rmtree(cache_dir)
