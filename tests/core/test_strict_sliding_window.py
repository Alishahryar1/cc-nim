"""Direct tests for :class:`core.rate_limit.StrictSlidingWindowLimiter`."""

import time

import pytest

from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter


@pytest.mark.asyncio
async def test_strict_window_allows_burst_then_blocks():
    lim = StrictSlidingWindowLimiter(rate_limit=2, rate_window=0.2)
    await lim.acquire()
    await lim.acquire()
    start = time.monotonic()
    await lim.acquire()
    assert time.monotonic() - start >= 0.15


@pytest.mark.asyncio
async def test_strict_window_async_context_manager():
    lim = StrictSlidingWindowLimiter(rate_limit=1, rate_window=0.15)

    async def run():
        async with lim:
            pass

    await run()
    start = time.monotonic()
    await run()
    assert time.monotonic() - start >= 0.1


def test_strict_window_rejects_invalid_config():
    with pytest.raises(ValueError):
        StrictSlidingWindowLimiter(rate_limit=0, rate_window=1.0)
    with pytest.raises(ValueError):
        StrictSlidingWindowLimiter(rate_limit=1, rate_window=0.0)
