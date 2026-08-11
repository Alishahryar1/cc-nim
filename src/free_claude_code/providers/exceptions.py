"""Provider exception classes compatible with ExecutionFailure semantics."""

from free_claude_code.core.failures import ExecutionFailure, FailureKind


class APIError(ExecutionFailure):
    def __init__(
        self,
        message: str = "API error",
        status_code: int = 500,
        raw_error: str | None = None,
    ) -> None:
        super().__init__(
            kind=FailureKind.UPSTREAM,
            status_code=status_code,
            message=message,
            retryable=False,
        )
        self.raw_error = raw_error


class AuthenticationError(ExecutionFailure):
    def __init__(
        self, message: str = "Authentication failed", raw_error: str | None = None
    ) -> None:
        super().__init__(
            kind=FailureKind.AUTHENTICATION,
            status_code=401,
            message=message,
            retryable=False,
        )
        self.raw_error = raw_error


class InvalidRequestError(ExecutionFailure):
    def __init__(
        self, message: str = "Invalid request", raw_error: str | None = None
    ) -> None:
        super().__init__(
            kind=FailureKind.INVALID_REQUEST,
            status_code=400,
            message=message,
            retryable=False,
        )
        self.raw_error = raw_error


class RateLimitError(ExecutionFailure):
    def __init__(
        self, message: str = "Rate limit reached", raw_error: str | None = None
    ) -> None:
        super().__init__(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message=message,
            retryable=True,
        )
        self.raw_error = raw_error


class OverloadedError(ExecutionFailure):
    def __init__(
        self, message: str = "Provider overloaded", raw_error: str | None = None
    ) -> None:
        super().__init__(
            kind=FailureKind.OVERLOADED,
            status_code=529,
            message=message,
            retryable=True,
        )
        self.raw_error = raw_error
