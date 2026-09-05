"""Worker boundary for managed configuration; runtime state stays on the loop."""

from collections.abc import Mapping

from anyio import to_thread

from free_claude_code.config.admin.persistence import (
    PreparedAdminUpdate,
    prepare_admin_update,
)
from free_claude_code.config.admin.state import ConfigInputValue, ValueState
from free_claude_code.config.admin.values import load_config_response, load_value_state
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject


class ConfigurationService:
    def __init__(self, store: ManagedConfigStore) -> None:
        self._store = store

    async def initialize(self) -> None:
        await to_thread.run_sync(self._store.initialize)

    async def admin_config(self) -> JsonObject:
        return load_config_response(await to_thread.run_sync(self._store.read))

    async def admin_values(self) -> ValueState:
        return load_value_state(await to_thread.run_sync(self._store.read))

    async def prepare(
        self, updates: Mapping[str, ConfigInputValue], active_settings: Settings
    ) -> PreparedAdminUpdate:
        snapshot = await to_thread.run_sync(self._store.read)
        return prepare_admin_update(updates, snapshot, active_settings)

    async def commit(self, prepared: PreparedAdminUpdate) -> JsonObject:
        if not prepared.valid:
            raise ValueError("Cannot commit an invalid Admin update")
        await to_thread.run_sync(self._store.commit, prepared.target_values)
        return prepared.applied_response()
