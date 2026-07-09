import asyncio
import contextlib
import time

import pytest
import pytest_asyncio

from free_claude_code.messaging.limiter import MessagingRateLimiter


class TestMessagingRateLimiter:
    """Tests for MessagingRateLimiter."""

    @pytest_asyncio.fixture(autouse=True)
    async def reset_limiter(self):
        """Reset singleton before each test."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)

        yield

        await MessagingRateLimiter.shutdown_instance(timeout=0.1)

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that get_instance returns the same object."""
        limiter1 = await MessagingRateLimiter.get_instance(
            rate_limit=1, rate_window=0.5
        )
        limiter2 = await MessagingRateLimiter.get_instance(
            rate_limit=99, rate_window=99.0
        )
        assert limiter1 is limiter2
        # First-construction wins for rate parameters
        assert limiter1.limiter._rate_limit == 1
        assert limiter1.limiter._rate_window == 0.5

    @pytest.mark.asyncio
    async def test_compaction(self):
        """
        Verify multiple rapid requests with same dedup_key are compacted.
        Logic ported from verify_limiter.py
        """
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        call_counts = {}

        async def mock_edit(msg_id, content):
            call_counts[msg_id] = call_counts.get(msg_id, 0) + 1
            return f"done_{content}"

        # Spam 5 edits
        for i in range(5):
            limiter.fire_and_forget(
                lambda i=i: mock_edit("msg1", f"update_{i}"), dedup_key="edit:msg1"
            )

        # Wait for processing
        # 1st might go through immediately, subsequent ones queue and compact
        await asyncio.sleep(2.5)

        # Expected: ~2 calls (first and last)
        assert call_counts["msg1"] <= 2, (
            f"Expected compaction to reduce calls, but got {call_counts.get('msg1', 0)}"
        )
        assert call_counts["msg1"] >= 1, "Expected at least one call"

    @pytest.mark.asyncio
    async def test_compaction_and_futures_resolution(self):
        """
        Verify that even when compacted, all futures resolve to the result of the LAST execution.
        Logic ported from verify_limiter_v2.py
        """
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=0.5)

        call_counts = {}
        msg_id = "test_msg_hang"

        async def mock_edit(mid, content):
            call_counts[mid] = call_counts.get(mid, 0) + 1
            await asyncio.sleep(0.05)
            return f"result_{content}"

        async def task(i):
            return await limiter.enqueue(
                lambda i=i: mock_edit(msg_id, f"v{i}"), dedup_key=f"edit:{msg_id}"
            )

        start_time = time.time()

        # Enqueue 3 tasks concurrently
        results = await asyncio.gather(task(1), task(2), task(3))

        duration = time.time() - start_time

        # All results should be the LAST one executed
        for res in results:
            assert res == "result_v3", f"Expected result_v3, got {res}"

        # Should be reasonably fast
        assert duration < 2.0, "Execution took too long"

        # Calls should be compacted
        assert call_counts[msg_id] <= 2, f"Too many actual calls: {call_counts[msg_id]}"

    @pytest.mark.asyncio
    async def test_flood_wait_handling(self):
        """Test that FloodWait exceptions pause the worker."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        # Mock exception with .seconds attribute
        class FloodWait(Exception):
            def __init__(self, seconds):
                self.seconds = seconds
                super().__init__(f"Flood wait {seconds}s")

        call_count = 0

        async def mock_fail():
            nonlocal call_count
            call_count += 1
            raise FloodWait(1)  # 1 second wait

        async def mock_success():
            nonlocal call_count
            call_count += 1
            return "success"

        # First call fails and triggers pause
        with contextlib.suppress(Exception):
            await limiter.enqueue(mock_fail, dedup_key="key1")

        assert limiter._paused_until > 0

        # Enqueue success, it should wait
        start = time.time()
        await limiter.enqueue(mock_success, dedup_key="key2")
        duration = time.time() - start

        # Should have waited at least ~1s
        assert duration >= 0.9, (
            f"Should have waited for FloodWait, but took {duration:.2f}s"
        )
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_flood_wait_retry_after_parsing(self):
        """Error message with 'retry after N' parses the wait seconds."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        async def mock_flood():
            raise Exception("Flood wait: retry after 2 seconds")

        with contextlib.suppress(Exception):
            await limiter.enqueue(mock_flood, dedup_key="retry_parse")

        # Should have parsed "after 2" -> 2 seconds
        assert limiter._paused_until > 0

    @pytest.mark.asyncio
    async def test_non_flood_exception_no_pause(self):
        """Non-flood exception doesn't trigger pause."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        async def mock_error():
            raise ValueError("some regular error")

        with contextlib.suppress(ValueError):
            await limiter.enqueue(mock_error, dedup_key="non_flood")

        # Should NOT have paused since it's not a flood error
        assert limiter._paused_until == 0

    @pytest.mark.asyncio
    async def test_flood_with_seconds_attribute(self):
        """Exception with .seconds attribute uses that value for pause."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        class FloodWaitCustom(Exception):
            def __init__(self):
                self.seconds = 2
                super().__init__("Flood wait custom")

        async def mock_flood():
            raise FloodWaitCustom()

        with contextlib.suppress(Exception):
            await limiter.enqueue(mock_flood, dedup_key="flood_sec")

        assert limiter._paused_until > 0

    @pytest.mark.asyncio
    async def test_proactive_strict_sliding_window(self):
        """
        Proactive limiter should enforce a strict sliding window:
        for any i, t[i+rate_limit] - t[i] >= rate_window (within tolerance).
        """
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=2, rate_window=0.5)

        async def acquire(i: int) -> float:
            async def _do() -> float:
                return time.monotonic()

            return await limiter.enqueue(_do, dedup_key=f"strict:{i}")

        acquired = await asyncio.gather(*(acquire(i) for i in range(5)))
        acquired.sort()

        rate_limit = 2
        rate_window = 0.5
        tolerance = 0.05
        for i in range(len(acquired) - rate_limit):
            assert acquired[i + rate_limit] - acquired[i] >= rate_window - tolerance, (
                f"Sliding window violated at i={i}: "
                f"dt={acquired[i + rate_limit] - acquired[i]:.3f}s"
            )

    @pytest.mark.asyncio
    async def test_compaction_last_task_fails_all_futures_get_exception(self):
        """When compacted task's last func fails, all futures get the exception."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        async def ok_task():
            return "ok"

        async def fail_task():
            raise RuntimeError("last task failed")

        future1 = asyncio.create_task(limiter.enqueue(ok_task, dedup_key="fail_key"))
        future2 = asyncio.create_task(limiter.enqueue(fail_task, dedup_key="fail_key"))

        with pytest.raises(RuntimeError, match="last task failed"):
            await future1
        with pytest.raises(RuntimeError, match="last task failed"):
            await future2

    @pytest.mark.asyncio
    async def test_fire_and_forget_failure_logged(self, caplog):
        """fire_and_forget with failing task logs error and does not re-raise."""
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        limiter = await MessagingRateLimiter.get_instance(rate_limit=1, rate_window=1.0)

        async def fail_task():
            raise ValueError("fire_and_forget failed")

        limiter.fire_and_forget(fail_task, dedup_key="fire_fail")
        await asyncio.sleep(1.5)

        joined = " ".join(str(r.message) for r in caplog.records)
        assert "ValueError" in joined
        assert "fire_and_forget failed" not in joined
