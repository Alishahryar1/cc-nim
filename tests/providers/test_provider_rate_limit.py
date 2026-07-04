import asyncio
import time

import openai
import pytest
import pytest_asyncio
from httpx import Request

from providers.rate_limit import (
    DEFAULT_UPSTREAM_MAX_RETRIES,
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    GlobalRateLimiter,
    retryable_upstream_status,
)


def test_upstream_transient_retry_total_attempts_is_five() -> None:
    assert UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS == 5
    assert DEFAULT_UPSTREAM_MAX_RETRIES == 4


def _statusless_api_error(message: str, body: object | None) -> openai.APIError:
    return openai.APIError(message, request=Request("POST", "http://x"), body=body)


def test_retryable_upstream_status_reads_statusless_api_error_body_status() -> None:
    exc = _statusless_api_error(
        "stream embedded error",
        {"error": {"message": "internal failure", "code": 500}},
    )

    assert retryable_upstream_status(exc) == 500


def test_retryable_upstream_status_reads_statusless_resource_exhausted_text() -> None:
    exc = _statusless_api_error(
        "ResourceExhausted: limit reached while generating response",
        {"error": {"message": "ResourceExhausted: limit reached"}},
    )

    assert retryable_upstream_status(exc) == 503


class TestProviderRateLimiter:
    """Tests for providers.rate_limit.GlobalRateLimiter."""

    @pytest_asyncio.fixture(autouse=True)
    async def reset_limiter(self):
        """Reset singleton before each test."""
        GlobalRateLimiter.reset_instance()
        yield
        GlobalRateLimiter.reset_instance()

    @pytest.mark.asyncio
    async def test_proactive_throttling(self):
        """
        Test proactive throttling.
        Logic ported from verify_provider_limiter.py
        """
        # Re-init with tight limits: 1 request per 0.25 second
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=1, rate_window=0.25)

        start_time = time.monotonic()

        async def call_limiter():
            await limiter.wait_if_blocked()
            return time.monotonic()

        # 5 requests.
        # R0 -> 0s
        # R1 -> 0.25s
        # R2 -> 0.50s
        # R3 -> 0.75s
        # R4 -> 1.00s
        results = [await call_limiter() for _ in range(5)]

        total_time = time.monotonic() - start_time

        assert len(results) == 5
        # Should take at least ~1.0s
        assert total_time >= 0.9, f"Throttling failed, took too fast: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_reactive_blocking(self):
        """
        Test reactive blocking when set_blocked is called.
        Logic ported from verify_provider_limiter.py
        """
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance()

        start_time = time.monotonic()

        # Manually block for 1.5s
        block_time = 1.5
        limiter.set_blocked(block_time)

        assert limiter.is_blocked()

        async def call_limiter():
            return await limiter.wait_if_blocked()

        # Run 2 calls concurrently
        # They should both wait for the block time
        results = await asyncio.gather(call_limiter(), call_limiter())

        total_time = time.monotonic() - start_time

        # Both should report having waited reactively
        assert all(results) is True
        assert total_time >= block_time - 0.1, (
            f"Reactive block failed, took {total_time:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_set_blocked_zero_immediately_unblocks(self):
        """set_blocked(0) should not actually block."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        limiter.set_blocked(0)

        # Should not be blocked since 0 seconds from now is already past
        await asyncio.sleep(0.01)
        assert limiter.is_blocked() is False
        assert limiter.remaining_wait() == 0

    @pytest.mark.asyncio
    async def test_remaining_wait_when_not_blocked(self):
        """remaining_wait() should return 0 when not blocked."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        assert limiter.remaining_wait() == 0

    @pytest.mark.asyncio
    async def test_remaining_wait_decreases(self):
        """remaining_wait() should decrease over time."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        limiter.set_blocked(2.0)

        wait1 = limiter.remaining_wait()
        assert wait1 > 1.5

        await asyncio.sleep(0.5)
        wait2 = limiter.remaining_wait()
        assert wait2 < wait1

    @pytest.mark.asyncio
    async def test_is_blocked_false_initially(self):
        """is_blocked() should be False for a fresh limiter."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        assert limiter.is_blocked() is False

    @pytest.mark.asyncio
    async def test_high_rate_limit_no_throttling(self):
        """Very high rate limit should not cause throttling."""
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=10000, rate_window=60)

        start = time.monotonic()
        for _ in range(20):
            await limiter.wait_if_blocked()
        duration = time.monotonic() - start

        # 20 requests with 10000 limit should be near-instant
        assert duration < 1.0, f"High rate limit caused throttling: {duration:.2f}s"

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """get_instance should return the same object."""
        limiter1 = GlobalRateLimiter.get_instance(rate_limit=10, rate_window=1)
        limiter2 = GlobalRateLimiter.get_instance()
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_reset_instance(self):
        """reset_instance should allow creating a new instance."""
        limiter1 = GlobalRateLimiter.get_instance(rate_limit=10, rate_window=1)
        GlobalRateLimiter.reset_instance()
        limiter2 = GlobalRateLimiter.get_instance(rate_limit=20, rate_window=2)
        assert limiter1 is not limiter2

    @pytest.mark.asyncio
    async def test_wait_if_blocked_returns_false_when_not_blocked(self):
        """wait_if_blocked should return False when not reactively blocked."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        result = await limiter.wait_if_blocked()
        assert result is False

    @pytest.mark.asyncio
    async def test_proactive_strict_rolling_window(self):
        """
        Proactive limiter should enforce a strict rolling window:
        for any i, t[i+rate_limit] - t[i] >= rate_window (within tolerance).
        """
        GlobalRateLimiter.reset_instance()
        rate_limit = 2
        rate_window = 0.5
        limiter = GlobalRateLimiter.get_instance(
            rate_limit=rate_limit, rate_window=rate_window
        )

        acquired: list[float] = []

        async def acquire():
            await limiter.wait_if_blocked()
            acquired.append(time.monotonic())

        # Trigger concurrency; without strict rolling windows, this can burst.
        await asyncio.gather(*(acquire() for _ in range(5)))

        acquired.sort()
        assert len(acquired) == 5

        tolerance = 0.05
        for i in range(len(acquired) - rate_limit):
            assert acquired[i + rate_limit] - acquired[i] >= rate_window - tolerance, (
                f"Rolling window violated at i={i}: "
                f"dt={acquired[i + rate_limit] - acquired[i]:.3f}s"
            )

    @pytest.mark.asyncio
    async def test_init_rate_limit_zero_raises(self):
        """rate_limit <= 0 raises ValueError."""
        GlobalRateLimiter.reset_instance()
        with pytest.raises(ValueError, match="rate_limit must be > 0"):
            GlobalRateLimiter(rate_limit=0, rate_window=60)

    @pytest.mark.asyncio
    async def test_init_rate_window_zero_raises(self):
        """rate_window <= 0 raises ValueError."""
        GlobalRateLimiter.reset_instance()
        with pytest.raises(ValueError, match="rate_window must be > 0"):
            GlobalRateLimiter(rate_limit=10, rate_window=0)

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhaust_retries_raises(self):
        """When all 429 retries exhausted, last exception is raised."""
        import openai
        from httpx import Request, Response

        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        def make_429():
            return openai.RateLimitError(
                "rate limited",
                response=Response(429, request=Request("POST", "http://x")),
                body={},
            )

        async def fail():
            raise make_429()

        with pytest.raises(openai.RateLimitError):
            await limiter.execute_with_retry(
                fail, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
            )

    @pytest.mark.asyncio
    async def test_execute_with_retry_succeeds_on_retry(self):
        """429 then success returns result."""
        import openai
        from httpx import Request, Response

        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        def make_429():
            return openai.RateLimitError(
                "rate limited",
                response=Response(429, request=Request("POST", "http://x")),
                body={},
            )

        call_count = 0

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise make_429()
            return "ok"

        result = await limiter.execute_with_retry(
            fail_then_ok, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
        )
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_succeeds_on_httpx_429(self):
        """HTTP 429 as httpx.HTTPStatusError then success returns result."""
        import httpx
        from httpx import Request, Response

        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        call_count = 0

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                r = Response(429, request=Request("POST", "http://x"), text="slow")
                raise httpx.HTTPStatusError(
                    "Too Many Requests", request=r.request, response=r
                )
            return "ok"

        result = await limiter.execute_with_retry(
            fail_then_ok, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
        )
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_execute_with_retry_succeeds_on_openai_internal_server_error_5xx(
        self, status_code
    ):
        """5xx as openai.InternalServerError then success."""
        import openai
        from httpx import Request, Response

        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        def make_upstream_error():
            return openai.InternalServerError(
                "unavailable",
                response=Response(status_code, request=Request("POST", "http://x")),
                body={},
            )

        call_count = 0

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise make_upstream_error()
            return "ok"

        result = await limiter.execute_with_retry(
            fail_then_ok, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
        )
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_execute_with_retry_succeeds_on_httpx_5xx(self, status_code):
        """HTTP 5xx as httpx.HTTPStatusError then success."""
        import httpx
        from httpx import Request, Response

        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        call_count = 0

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                r = Response(
                    status_code, request=Request("POST", "http://x"), text="error"
                )
                raise httpx.HTTPStatusError(
                    "Server Error", request=r.request, response=r
                )
            return "ok"

        result = await limiter.execute_with_retry(
            fail_then_ok, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
        )
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_succeeds_on_statusless_transient_api_error(self):
        """Status-less SDK APIError transient markers participate in backoff retry."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        call_count = 0

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _statusless_api_error(
                    "ResourceExhausted: limit reached while generating response",
                    {"error": {"message": "ResourceExhausted: limit reached"}},
                )
            return "ok"

        result = await limiter.execute_with_retry(
            fail_then_ok, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
        )

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_execute_with_retry_exhaust_openai_5xx_raises(self, status_code):
        """When all 5xx retries exhausted (OpenAI SDK), last InternalServerError is raised."""
        import openai
        from httpx import Request, Response

        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        exc = openai.InternalServerError(
            "unavailable",
            response=Response(status_code, request=Request("POST", "http://x")),
            body={},
        )

        async def always_fail():
            raise exc

        with pytest.raises(openai.InternalServerError):
            await limiter.execute_with_retry(
                always_fail, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
            )

    @pytest.mark.asyncio
    async def test_execute_with_retry_httpx_400_raises_immediately(self):
        """Non-retryable 4xx is not wrapped by execute_with_retry loop."""
        import httpx
        from httpx import Request, Response

        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)

        call_count = 0

        async def bad_request():
            nonlocal call_count
            call_count += 1
            r = Response(400, request=Request("POST", "http://x"), text="bad request")
            raise httpx.HTTPStatusError("Bad Request", request=r.request, response=r)

        with pytest.raises(httpx.HTTPStatusError):
            await limiter.execute_with_retry(
                bad_request,
                max_retries=2,
                base_delay=0.01,
                max_delay=0.1,
                jitter=0,
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_concurrency_zero_raises(self):
        """max_concurrency <= 0 raises ValueError."""
        GlobalRateLimiter.reset_instance()
        with pytest.raises(ValueError, match="max_concurrency must be > 0"):
            GlobalRateLimiter(rate_limit=10, rate_window=60, max_concurrency=0)

    @pytest.mark.asyncio
    async def test_concurrency_slot_limits_simultaneous_streams(self):
        """At most max_concurrency streams can hold a slot simultaneously."""
        GlobalRateLimiter.reset_instance()
        max_concurrency = 2
        limiter = GlobalRateLimiter.get_instance(
            rate_limit=100, rate_window=60, max_concurrency=max_concurrency
        )

        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def stream_task(hold_time: float) -> None:
            nonlocal peak_concurrent, current_concurrent
            async with limiter.concurrency_slot():
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > peak_concurrent:
                        peak_concurrent = current_concurrent
                await asyncio.sleep(hold_time)
                async with lock:
                    current_concurrent -= 1

        # Launch 5 tasks that each hold the slot; only 2 can be active at once
        await asyncio.gather(*(stream_task(0.05) for _ in range(5)))

        assert peak_concurrent <= max_concurrency, (
            f"Concurrency exceeded: peak={peak_concurrent}, max={max_concurrency}"
        )

    @pytest.mark.asyncio
    async def test_concurrency_slot_releases_on_exception(self):
        """Slot is released even when the body raises an exception."""
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(
            rate_limit=100, rate_window=60, max_concurrency=1
        )
        assert limiter._concurrency_sem is not None

        with pytest.raises(RuntimeError):
            async with limiter.concurrency_slot():
                raise RuntimeError("boom")

        # Semaphore value should be restored (1 available again)
        assert limiter._concurrency_sem._value == 1

    @pytest.mark.asyncio
    async def test_get_instance_passes_max_concurrency(self):
        """get_instance forwards max_concurrency to the singleton."""
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(
            rate_limit=10, rate_window=60, max_concurrency=3
        )
        assert limiter._concurrency_sem is not None
        assert limiter._concurrency_sem._value == 3

    @pytest.mark.asyncio
    async def test_scoped_instances_are_isolated(self):
        """Provider-scoped limiters do not share reactive block state."""
        GlobalRateLimiter.reset_instance()
        nim = GlobalRateLimiter.get_scoped_instance(
            "nvidia_nim", rate_limit=10, rate_window=60
        )
        openrouter = GlobalRateLimiter.get_scoped_instance(
            "open_router", rate_limit=20, rate_window=30
        )

        assert nim is not openrouter
        nim.set_blocked(1.0)

        assert nim.is_blocked() is True
        assert openrouter.is_blocked() is False
