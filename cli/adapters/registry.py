"""Internal client CLI adapter registry."""

from __future__ import annotations

from .base import ClientCliAdapter
from .claude import CLAUDE_CLI_ADAPTER
from .codex import CODEX_CLI_ADAPTER

DEFAULT_CLIENT_CLI_ID = "claude"

_ADAPTERS: dict[str, ClientCliAdapter] = {
    CLAUDE_CLI_ADAPTER.id: CLAUDE_CLI_ADAPTER,
    CODEX_CLI_ADAPTER.id: CODEX_CLI_ADAPTER,
}


def get_client_cli_adapter(
    client_cli_id: str = DEFAULT_CLIENT_CLI_ID,
) -> ClientCliAdapter:
    """Return a registered client CLI adapter by id."""

    try:
        return _ADAPTERS[client_cli_id]
    except KeyError as exc:
        raise ValueError(f"Unknown client CLI adapter: {client_cli_id}") from exc
