"""Harness-owned initial identities and native launch commands."""

import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from free_claude_code.application.browser_sessions import (
    BROWSER_SESSION_HEADER,
    BrowserSessionHarness,
    BrowserSessionUnavailableError,
    HarnessAvailability,
)
from free_claude_code.cli.launchers.claude import claude_binary_name
from free_claude_code.cli.launchers.codex import codex_binary_name


class HarnessDriver:
    """Own one harness's executable discovery, initial ID, and argv contract."""

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

    def initial_native_id(self) -> str | None:
        """Return an FCC-assigned ID, or defer native identity to the harness."""

        return str(uuid4())

    def command(
        self,
        native_id: str | None,
        *,
        started_once: bool,
        binding_token: str | None = None,
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
        native_id: str | None,
        *,
        started_once: bool,
        binding_token: str | None = None,
    ) -> list[str]:
        if native_id is None:
            raise BrowserSessionUnavailableError("Claude session identity is missing.")
        flag = "--resume" if started_once else "--session-id"
        return [self._wrapper(), flag, native_id]


class PiDriver(HarnessDriver):
    harness = BrowserSessionHarness.PI
    wrapper_name = "fcc-pi"
    client_name = "pi"

    def command(
        self,
        native_id: str | None,
        *,
        started_once: bool,
        binding_token: str | None = None,
    ) -> list[str]:
        if native_id is None:
            raise BrowserSessionUnavailableError("Pi session identity is missing.")
        return [self._wrapper(), "--session-id", native_id]


class CodexDriver(HarnessDriver):
    harness = BrowserSessionHarness.CODEX
    wrapper_name = "fcc-codex"
    client_name = codex_binary_name()

    def initial_native_id(self) -> None:
        """Let the native Codex CLI create and persist its own thread."""

        return None

    def command(
        self,
        native_id: str | None,
        *,
        started_once: bool,
        binding_token: str | None = None,
    ) -> list[str]:
        if binding_token is None:
            raise BrowserSessionUnavailableError(
                "Codex browser-session correlation is unavailable."
            )
        command = [
            self._wrapper(),
            "-c",
            _toml_assignment(
                f"model_providers.fcc.http_headers.{BROWSER_SESSION_HEADER}",
                binding_token,
            ),
        ]
        if native_id is not None:
            command.extend(["resume", native_id])
        return command


class HarnessDriverRegistry:
    """Fixed registry for the three customer-facing native harnesses."""

    def __init__(self, drivers: Sequence[HarnessDriver] | None = None) -> None:
        selected = drivers or (ClaudeDriver(), CodexDriver(), PiDriver())
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


def _toml_assignment(key: str, value: str) -> str:
    return f"{key}={json.dumps(value)}"
