import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import psutil
import pytest

from free_claude_code.application.terminal import TerminalClientPort
from free_claude_code.runtime import terminal_zellij


class _UnusedClientFactory:
    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> TerminalClientPort:
        raise AssertionError((argv, cwd, env, rows, columns))


class _UnusedContainment:
    async def establish(self, server: psutil.Process) -> None:
        raise AssertionError(server)

    async def terminate(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_zellij_host_uses_only_its_managed_paths(tmp_path: Path) -> None:
    binary = tmp_path / "managed" / "zellij"
    config = tmp_path / "runtime" / "config.kdl"
    sockets = tmp_path / "sockets"
    data = tmp_path / "runtime" / "data"
    host = terminal_zellij.ZellijTerminalEngineHost(
        binary=binary,
        config=config,
        sockets=sockets,
        data=data,
        lock_path=tmp_path / "terminal.lock",
        client_factory=_UnusedClientFactory(),
        containment_factory=_UnusedContainment,
    )

    assert host.command_prefix == (
        str(binary),
        "--data-dir",
        str(data),
        "--config",
        str(config),
    )
    assert host._environment({"UNCHANGED": "yes"}) == {
        "UNCHANGED": "yes",
        "ZELLIJ_SOCKET_DIR": str(sockets),
        "ZELLIJ_CONFIG_FILE": str(config),
    }


def test_zellij_config_disables_ui_plugins_web_and_resurrection() -> None:
    config = terminal_zellij._CONFIG

    for contract in (
        "keybinds clear-defaults=true",
        'default_mode "locked"',
        "pane_frames false",
        "mirror_session true",
        "session_serialization false",
        "serialize_pane_viewport false",
        "disable_session_metadata true",
        "web_server false",
        'web_sharing "disabled"',
        "scroll_buffer_size 10000",
        "plugins {\n}",
        "load_plugins {\n}",
    ):
        assert contract in config


def test_snapshot_parser_preserves_ansi_scrollback_and_viewport() -> None:
    payload = json.dumps(
        {
            "event": "pane_update",
            "pane_id": "terminal_0",
            "is_initial": True,
            "scrollback": ["old", "\u001b[31mred\u001b[0m"],
            "viewport": ["current", "prompt>"],
        }
    ).encode()

    snapshot = terminal_zellij._parse_snapshot(payload)

    assert snapshot.scrollback == b"old\r\n\x1b[31mred\x1b[0m\r\n"
    assert snapshot.viewport == b"current\r\nprompt>\r\n"
    assert snapshot.rendered == snapshot.scrollback + snapshot.viewport


@pytest.mark.parametrize(
    "payload",
    (
        b"not-json",
        b"[]",
        b'{"event":"pane_update","pane_id":"other","is_initial":true,"viewport":[]}',
        b'{"event":"pane_update","pane_id":"terminal_0","is_initial":false,"viewport":[]}',
        b'{"event":"pane_update","pane_id":"terminal_0","is_initial":true,"viewport":"bad"}',
    ),
)
def test_snapshot_parser_rejects_non_contract_payloads(payload: bytes) -> None:
    with pytest.raises(RuntimeError, match="snapshot"):
        terminal_zellij._parse_snapshot(payload)


def test_pane_parser_requires_exactly_one_live_root_terminal() -> None:
    pane = {
        "id": 0,
        "is_plugin": False,
        "exited": False,
        "terminal_command": "/bin/sh",
    }

    assert (
        terminal_zellij._parse_single_terminal_pane(json.dumps([pane]).encode()) == pane
    )
    assert terminal_zellij._parse_single_terminal_pane(b"invalid") is None
    assert (
        terminal_zellij._parse_single_terminal_pane(json.dumps([pane, pane]).encode())
        is None
    )
    assert (
        terminal_zellij._parse_single_terminal_pane(
            json.dumps([{**pane, "exited": True}]).encode()
        )
        is None
    )
    assert (
        terminal_zellij._parse_single_terminal_pane(
            json.dumps([{**pane, "is_plugin": True}]).encode()
        )
        is None
    )


def test_launch_command_preserves_argv_without_a_shell(tmp_path: Path) -> None:
    executable = Path(os.environ.get("COMSPEC", "/bin/sh"))
    environment = dict(os.environ)

    argv, expected = terminal_zellij._launch_command(
        (str(executable), "argument with spaces", "&&", "literal"),
        tmp_path,
        environment,
    )

    assert expected == terminal_zellij._resolved(executable)
    if os.name == "nt":
        assert argv == (executable.name, "argument with spaces", "&&", "literal")
        assert environment["PATH"].split(os.pathsep)[0] == str(expected.parent)
    else:
        assert argv == (str(executable), "argument with spaces", "&&", "literal")


def test_pinned_version_agrees_with_both_installers() -> None:
    root = Path(__file__).resolve().parents[2]
    powershell = (root / "scripts" / "install.ps1").read_text(encoding="utf-8")
    posix = (root / "scripts" / "install.sh").read_text(encoding="utf-8")

    assert f'$ZellijVersion = "{terminal_zellij.ZELLIJ_VERSION}"' in powershell
    assert f'ZELLIJ_VERSION="{terminal_zellij.ZELLIJ_VERSION}"' in posix


def test_zellij_license_is_packaged_with_admin_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    license_text = (
        root
        / "src"
        / "free_claude_code"
        / "api"
        / "admin_static"
        / "zellij-LICENSE.txt"
    ).read_text(encoding="utf-8")

    assert "Copyright (c) 2020 Zellij contributors" in license_text
    assert "MIT License" in license_text
