"""Client CLI adapter implementations."""

from .claude import CLAUDE_CLI_ADAPTER, ClaudeCliAdapter
from .codex import CODEX_CLI_ADAPTER, CodexCliAdapter
from .registry import DEFAULT_CLIENT_CLI_ID, get_client_cli_adapter

__all__ = [
    "CLAUDE_CLI_ADAPTER",
    "CODEX_CLI_ADAPTER",
    "DEFAULT_CLIENT_CLI_ID",
    "ClaudeCliAdapter",
    "CodexCliAdapter",
    "get_client_cli_adapter",
]
