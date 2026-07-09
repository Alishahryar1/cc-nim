"""Unified exception hierarchy for providers."""

from typing import Any


class ProviderError(Exception):
    """Base exception for all provider errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "api_error",
        raw_error: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.raw_error = raw_error

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic-compatible error response."""
        return {
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        }


class AuthenticationError(ProviderError):
    """Raised when API key is invalid or missing."""

    def __init__(self, message: str, raw_error: Any = None):
        super().__init__(
            message,
            status_code=401,
            error_type="authentication_error",
            raw_error=raw_error,
        )


class InvalidRequestError(ProviderError):
    """Raised when the request parameters are invalid."""

    def __init__(self, message: str, raw_error: Any = None):
        super().__init__(
            message,
            status_code=400,
            error_type="invalid_request_error",
            raw_error=raw_error,
        )


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, raw_error: Any = None):
        super().__init__(
            message,
            status_code=429,
            error_type="rate_limit_error",
            raw_error=raw_error,
        )


class OverloadedError(ProviderError):
    """Raised when the provider is overloaded."""

    def __init__(self, message: str, raw_error: Any = None):
        super().__init__(
            message,
            status_code=529,
            error_type="overloaded_error",
            raw_error=raw_error,
        )


class APIError(ProviderError):
    """Raised when the provider returns a generic API error."""

    def __init__(self, message: str, status_code: int = 500, raw_error: Any = None):
        super().__init__(
            message,
            status_code=status_code,
            error_type="api_error",
            raw_error=raw_error,
        )


class UnknownProviderTypeError(InvalidRequestError):
    """Raised when ``provider_id`` is not registered in the provider map."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ServiceUnavailableError(ProviderError):
    """Raised when the server is not ready (e.g. app lifespan did not wire state)."""

    def __init__(self, message: str, raw_error: Any = None):
        super().__init__(
            message,
            status_code=503,
            error_type="api_error",
            raw_error=raw_error,
        )


class ModelListResponseError(ServiceUnavailableError):
    """Raised when a provider model-list response cannot be parsed safely."""
