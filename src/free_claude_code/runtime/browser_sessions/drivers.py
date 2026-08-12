"""Harness-owned native launch commands."""

import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from free_claude_code.application.browser_sessions import (
    BrowserSessionHarness,
    BrowserSessionUnavailableError,
    HarnessAvailability,
)
from free_claude_code.cli.launchers.claude import claude_binary_name


class HarnessDriver:
    """Own one harness's executable discovery and argv contract."""

    harness: BrowserSessionHarness
    wrapper_name: str
    client_name: str

    def availability(self) -> HarnessAvailability:
        if _resolve_command(self.wrapper_name) is None:
            return HarnessAvailability(
                self.harness,
                False,
                f"Update Free Claude Code to install {self.wrapper_name}.",
            )
        if shutil.which(self.client_name) is None:
            return HarnessAvailability(
                self.harness,
                False,
                f"Install {self.client_name} to use this harness.",
            )
        return HarnessAvailability(self.harness, True)

    def command(
        self,
        session_id: str,
        *,
        started_once: bool,
    ) -> list[str]:
        raise NotImplementedError

    def _wrapper(self) -> str:
        wrapper = _resolve_command(self.wrapper_name)
        if wrapper is None:
            raise BrowserSessionUnavailableError(
                f"{self.wrapper_name} is not installed. Update Free Claude Code."
            )
        if shutil.which(self.client_name) is None:
            raise BrowserSessionUnavailableError(
                f"{self.client_name} is not installed."
            )
        return wrapper


class ClaudeDriver(HarnessDriver):
    harness = BrowserSessionHarness.CLAUDE
    wrapper_name = "fcc-claude"
    client_name = claude_binary_name()

    def command(
        self,
        session_id: str,
        *,
        started_once: bool,
    ) -> list[str]:
        flag = "--resume" if started_once else "--session-id"
        return [self._wrapper(), flag, session_id]


class PiDriver(HarnessDriver):
    harness = BrowserSessionHarness.PI
    wrapper_name = "fcc-pi"
    client_name = "pi"

    def command(
        self,
        session_id: str,
        *,
        started_once: bool,
    ) -> list[str]:
        return [self._wrapper(), "--session-id", session_id]


class HarnessDriverRegistry:
    """Fixed registry for the two browser-session harnesses."""

    def __init__(self, drivers: Sequence[HarnessDriver] | None = None) -> None:
        selected = drivers or (ClaudeDriver(), PiDriver())
        self._drivers = {driver.harness: driver for driver in selected}

    def driver(self, harness: BrowserSessionHarness) -> HarnessDriver:
        driver = self._drivers.get(harness)
        if driver is None:
            raise BrowserSessionUnavailableError(
                f"Harness {harness.value!r} is unavailable."
            )
        return driver

    def availability(self) -> tuple[HarnessAvailability, ...]:
        return tuple(
            self.driver(harness).availability() for harness in BrowserSessionHarness
        )


def terminal_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the inherited process environment with browser PTY capabilities."""

    env = dict(os.environ if base is None else base)
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    return env


def _resolve_command(name: str) -> str | None:
    if resolved := shutil.which(name):
        return resolved
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [scripts_dir / name]
    if os.name == "nt":
        candidates.insert(0, scripts_dir / f"{name}.exe")
    return next((str(path) for path in candidates if path.is_file()), None)
