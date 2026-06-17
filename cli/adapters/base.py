"""Shared contracts for client CLI subprocess adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CliTaskRequest:
    """A single prompt execution request for a managed client CLI process."""

    prompt: str
    session_id: str | None = None
    fork_session: bool = False


@dataclass(frozen=True, slots=True)
class CliInvocation:
    """Concrete subprocess invocation assembled by a client CLI adapter."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CliParseState:
    """Mutable line-parser state for a single client CLI process run."""

    log_raw_cli_diagnostics: bool = False
    session_id_extracted: bool = False


class ClientCliAdapter(Protocol):
    """Adapter boundary for client CLI command/env construction and output parsing."""

    id: str
    display_name: str
    default_binary: str
    install_hint: str
    trace_stage: str
    process_launch_event: str
    trace_source: str

    def build_task_invocation(
        self,
        *,
        config: Any,
        request: CliTaskRequest,
        base_env: Mapping[str, str],
    ) -> CliInvocation:
        """Build the subprocess invocation for a managed task run."""
        ...

    def parse_stdout_line(self, line: str, state: CliParseState) -> Iterable[Any]:
        """Parse one stdout line into parser-ready internal CLI events."""
        ...

    def extract_session_id(self, event: Any) -> str | None:
        """Extract a persistent client CLI session id from a parsed event."""
        ...

    def get_launcher_binary_name(self, settings: Any) -> str:
        """Return the configured executable name for a wrapper entrypoint."""
        ...

    def build_launcher_command(
        self,
        *,
        binary_path: str,
        argv: Iterable[str],
        settings: Any,
        proxy_root_url: str,
    ) -> list[str]:
        """Build the wrapper subprocess command for a client CLI launch."""
        ...

    def build_launcher_env(
        self,
        *,
        proxy_root_url: str,
        auth_token: str,
        base_env: Mapping[str, str],
    ) -> dict[str, str]:
        """Build environment variables for a wrapper-launched client CLI."""
        ...
