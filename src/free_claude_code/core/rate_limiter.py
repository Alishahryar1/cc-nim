# src/free_claude_code/core/rate_limiter.py
import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: list[float] = []

    @property
    def is_available(self) -> bool:
        current_time = time.time()
        # Remove expired timestamps
        self.timestamps = [
            t for t in self.timestamps if current_time - t < self.window_seconds
        ]
        return len(self.timestamps) < self.max_requests

    def acquire(self) -> bool:
        if self.is_available:
            self.timestamps.append(time.time())
            return True
        return False
