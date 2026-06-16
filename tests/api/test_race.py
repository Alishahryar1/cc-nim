import asyncio
from unittest.mock import MagicMock, patch

import pytest

from api.model_router import ResolvedModel
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings


class MockProvider:
    def __init__(self, delay=0):
        self.delay = delay
        self.call_count = 0

    async def stream_response(
        self, request, input_tokens=0, request_id=None, thinking_enabled=None
    ):
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        yield "event: message_start\ndata: {}\n\n"


@pytest.mark.asyncio
async def test_staggered_race_mode():
    settings = Settings()
    # Mock performance tracker to have a low avg_ttft for the primary
    with patch("api.services.performance_tracker") as mock_perf:
        mock_perf.get_metrics.return_value = MagicMock(avg_ttft=0.1, success_count=10)

        provider1 = MockProvider(delay=0.05)  # Fast
        provider2 = MockProvider(delay=0.01)  # Faster but staggered

        def provider_getter(provider_id):
            if provider_id == "p1":
                return provider1
            return provider2

        service = ClaudeProxyService(settings, provider_getter)

        # Mock router to return candidates for p1 and p2
        with patch.object(service._model_router, "resolve_candidates") as mock_resolve:
            mock_resolve.side_effect = lambda m: [
                ResolvedModel(
                    original_model=m,
                    provider_id=m,
                    provider_model=m,
                    provider_model_ref=f"{m}/{m}",
                    thinking_enabled=True,
                )
            ]

            req = MessagesRequest(
                model="race/p1,p2", messages=[Message(role="user", content="hi")]
            )

            stream = service._run_race_mode(req, ["p1", "p2"])
            chunks = [chunk async for chunk in stream]

            # Since p1 is fast (0.05s) and stagger delay is 0.1 * 1.2 = 0.12s,
            # p2 should NOT have been called if p1 yielded before 0.12s.
            assert provider1.call_count == 1
            assert provider2.call_count == 0
            assert len(chunks) > 0


@pytest.mark.asyncio
async def test_staggered_race_mode_failover():
    settings = Settings()
    with patch("api.services.performance_tracker") as mock_perf:
        mock_perf.get_metrics.return_value = MagicMock(avg_ttft=0.01, success_count=10)

        provider1 = MockProvider(delay=0.5)  # Slow
        provider2 = MockProvider(delay=0.01)  # Fast

        def provider_getter(provider_id):
            if provider_id == "p1":
                return provider1
            return provider2

        service = ClaudeProxyService(settings, provider_getter)

        with patch.object(service._model_router, "resolve_candidates") as mock_resolve:
            mock_resolve.side_effect = lambda m: [
                ResolvedModel(
                    original_model=m,
                    provider_id=m,
                    provider_model=m,
                    provider_model_ref=f"{m}/{m}",
                    thinking_enabled=True,
                )
            ]

            req = MessagesRequest(
                model="race/p1,p2", messages=[Message(role="user", content="hi")]
            )

            stream = service._run_race_mode(req, ["p1", "p2"])
            chunks = [chunk async for chunk in stream]

            # p1 is slow (0.5s), stagger is 0.012s. p2 should win.
            assert provider1.call_count == 1
            assert provider2.call_count == 1
            assert len(chunks) > 0
