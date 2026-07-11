"""Protocol-neutral execution failure semantics."""

from dataclasses import FrozenInstanceError, dataclass
from enum import StrEnum


class FailureKind(StrEnum):
    """Stable failure categories shared across execution and wire adapters."""

    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    UPSTREAM = "upstream"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True, eq=False)
class ExecutionFailure(Exception):
    """A finalized provider-execution failure independent of any wire protocol."""

    kind: FailureKind
    status_code: int
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __setattr__(self, name: str, value: object) -> None:
        # Exception machinery must be able to update __traceback__, __cause__,
        # and __context__ while semantic failure fields remain immutable.
        if name in self.__slots__ and hasattr(self, name):
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        super().__setattr__(name, value)
