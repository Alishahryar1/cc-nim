"""Compose the managed terminal engine for the current server platform."""

import os

from free_claude_code.application.terminal import TerminalEngineHostPort
from free_claude_code.config.paths import (
    terminal_binary_path,
    terminal_config_path,
    terminal_lock_path,
    terminal_runtime_dir_path,
    terminal_socket_dir_path,
)

from .terminal_zellij import ZellijTerminalEngineHost

if os.name == "nt":
    from .terminal_pty_windows import (
        WindowsProcessContainment as PlatformProcessContainment,
    )
    from .terminal_pty_windows import (
        WindowsTerminalClientFactory as PlatformTerminalClientFactory,
    )
else:
    from .terminal_pty_posix import (
        PosixProcessContainment as PlatformProcessContainment,
    )
    from .terminal_pty_posix import (
        PosixTerminalClientFactory as PlatformTerminalClientFactory,
    )


def create_terminal_engine_host() -> TerminalEngineHostPort:
    runtime = terminal_runtime_dir_path()
    return ZellijTerminalEngineHost(
        binary=terminal_binary_path(),
        config=terminal_config_path(),
        sockets=terminal_socket_dir_path(),
        data=runtime / "data",
        lock_path=terminal_lock_path(),
        client_factory=PlatformTerminalClientFactory(),
        containment_factory=PlatformProcessContainment,
    )
