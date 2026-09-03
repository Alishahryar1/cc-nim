"""Select the native PTY adapter for the current FCC server platform."""

import os

from free_claude_code.application.terminal import TerminalProcessFactoryPort

if os.name == "nt":
    from .terminal_pty_windows import (
        WindowsTerminalProcessFactory as PlatformTerminalProcessFactory,
    )
else:
    from .terminal_pty_posix import (
        PosixTerminalProcessFactory as PlatformTerminalProcessFactory,
    )


def create_terminal_process_factory() -> TerminalProcessFactoryPort:
    return PlatformTerminalProcessFactory()
