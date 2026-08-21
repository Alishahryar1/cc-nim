"""Runtime capabilities consumed by the HTTP API adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountStatus,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.application.ports import RequestRuntimePort, TaskController
from free_claude_code.config.admin.state import ConfigInputValue
from free_claude_code.core.json_types import JsonObject


class AdminRuntimePort(Protocol):
    """Runtime operations exposed by the local Admin API."""

    async def apply_admin_config(
        self, updates: Mapping[str, ConfigInputValue]
    ) -> JsonObject: ...

    def admin_status(self) -> JsonObject: ...

    async def test_provider(self, provider_id: str) -> JsonObject: ...

    async def refresh_models(self) -> ProviderModelRefreshResult: ...

    async def request_restart(self) -> None: ...

    async def connected_account_status(
        self, provider_id: str
    ) -> ConnectedAccountStatus: ...

    async def start_connected_account_login(
        self,
        provider_id: str,
        mode: ConnectedAccountLoginMode,
    ) -> ConnectedAccountStatus: ...

    async def cancel_connected_account_login(
        self, provider_id: str
    ) -> ConnectedAccountStatus: ...

    async def disconnect_connected_account(
        self, provider_id: str
    ) -> ConnectedAccountStatus: ...


@dataclass(frozen=True, slots=True)
class ApiServices:
    """Complete runtime boundary required to construct the API application."""

    requests: RequestRuntimePort
    admin: AdminRuntimePort
    tasks: TaskController
