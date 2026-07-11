"""Deterministic application and readiness errors."""

from free_claude_code.core.failures import FailureKind


class ApplicationError(Exception):
    """Base for request/readiness failures, not finalized upstream failures."""

    kind: FailureKind
    status_code: int

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(ApplicationError):
    """The accepted request cannot be executed deterministically."""

    kind = FailureKind.INVALID_REQUEST
    status_code = 400


class UnknownProviderError(InvalidRequestError):
    """The configured provider identifier is not registered."""


class ApplicationUnavailableError(ApplicationError):
    """The application cannot currently provide a request runtime."""

    kind = FailureKind.UNAVAILABLE
    status_code = 503
