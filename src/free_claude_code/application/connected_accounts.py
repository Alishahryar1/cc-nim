"""Safe application boundary for provider connected accounts."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol


class ConnectedAccountLoginMode(StrEnum):
    """Supported interactive authorization flows."""

    BROWSER = "browser"
    DEVICE = "device"


class ConnectedAccountState(StrEnum):
    """Customer-visible connected-account lifecycle states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ConnectedAccountStatus:
    """Credential-free state safe to return from the local Admin API."""

    provider_id: str
    state: ConnectedAccountState
    connected: bool
    revision: int
    attempt_id: str | None = None
    email: str | None = None
    mode: ConnectedAccountLoginMode | None = None
    authorization_url: str | None = None
    verification_url: str | None = None
    user_code: str | None = None
    expires_at: int | None = None
    model_count: int | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize only the explicitly safe status contract."""

        return {key: value for key, value in asdict(self).items() if value is not None}


class ConnectedAccountPort(Protocol):
    """One process-lifetime provider authentication owner."""

    def is_connected(self) -> bool: ...

    def status(self) -> ConnectedAccountStatus: ...

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus: ...

    async def cancel_login(self) -> ConnectedAccountStatus: ...

    async def disconnect(self) -> ConnectedAccountStatus: ...

    async def close(self) -> None: ...
