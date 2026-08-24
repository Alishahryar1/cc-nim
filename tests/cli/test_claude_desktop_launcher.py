"""Tests for the ``fcc-claude-desktop`` console-script shim.

The shim delegates to :mod:`free_claude_code.cli.claude_desktop`; these
tests pin its CLI surface and verify it forwards to the shared routing
implementation (patched there for determinism).
"""

import json
from pathlib import Path

import pytest

from free_claude_code.cli import claude_desktop as routing
from free_claude_code.cli.launchers import claude_desktop
from free_claude_code.config.settings import Settings

_DETERMINISTIC_URL = "https://localhost:8443/claude-desktop"


@pytest.fixture(autouse=True)
def deterministic_gateway_url(monkeypatch: pytest.MonkeyPatch):
    """Keep gateway URL resolution off the network in every test."""

    monkeypatch.setattr(
        routing,
        "desktop_gateway_base_url",
        lambda settings: _DETERMINISTIC_URL,
    )


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    return tmp_path / "claude_desktop_config.json"


def _managed_block() -> dict[str, object]:
    return routing.fcc_managed_block(
        Settings(host="127.0.0.1", port=8082),
        gateway_base_url=_DETERMINISTIC_URL,
    )


def test_configure_applies_block_when_file_is_missing(fake_config: Path) -> None:
    assert not fake_config.exists()

    changed = claude_desktop.configure_claude_desktop_config(fake_config)

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
    assert data["inference"]["provider"] == "gateway"
    assert data["inference"]["inferenceGatewayBaseUrl"] == _DETERMINISTIC_URL


def test_configure_is_idempotent(fake_config: Path) -> None:
    assert claude_desktop.configure_claude_desktop_config(fake_config) is True

    # second call observes no diff.
    second = claude_desktop.configure_claude_desktop_config(fake_config)
    assert second is False


def test_configure_preserves_unrelated_keys(fake_config: Path) -> None:
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "mcpServers": {"fs": {"command": "node"}},
                "coworkUserFilesPath": "/home/user/cowork",
                "env": {"ANTHROPIC_BASE_URL": "https://localhost:8443"},
            }
        ),
        encoding="utf-8",
    )

    claude_desktop.configure_claude_desktop_config(fake_config)

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["preferences"] == {"theme": "dark"}
    assert data["mcpServers"] == {"fs": {"command": "node"}}
    assert data["coworkUserFilesPath"] == "/home/user/cowork"
    assert data["env"] == {"ANTHROPIC_BASE_URL": "https://localhost:8443"}
    assert data["modelDiscoveryEnabled"] is True


def test_configure_overwrites_partial_inference_block(fake_config: Path) -> None:
    fake_config.write_text(
        json.dumps(
            {"inference": {"provider": "anthropic", "credentialKind": "static"}}
        ),
        encoding="utf-8",
    )

    claude_desktop.configure_claude_desktop_config(fake_config)

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["inference"]["provider"] == "gateway"
    assert data["inference"]["credentialKind"] == "static"
    assert data["inference"]["inferenceGatewayBaseUrl"] == _DETERMINISTIC_URL


def test_unconfigure_drops_only_fcc_keys(fake_config: Path) -> None:
    fake_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "modelDiscoveryEnabled": True,
                "inference": _managed_block() | {"extra_user_key": "stay"},
            }
        ),
        encoding="utf-8",
    )

    changed = claude_desktop.unconfigure_claude_desktop_config(fake_config)

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in data
    assert data["inference"] == {"extra_user_key": "stay"}
    assert data["preferences"] == {"theme": "dark"}


def test_unconfigure_removes_entire_inference_block_when_only_fcc_keys(
    fake_config,
) -> None:
    claude_desktop.configure_claude_desktop_config(fake_config)
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    # Second pass is a no-op.
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is False


def test_unconfigure_silent_on_missing_file(tmp_path: Path) -> None:
    assert (
        claude_desktop.unconfigure_claude_desktop_config(tmp_path / "absent.json")
        is False
    )


def test_config_path_resolver_returns_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    for plat, expected_substring in (
        ("linux", "claude_desktop_config.json"),
        ("darwin", "claude_desktop_config.json"),
        ("win32", "claude_desktop_config.json"),
    ):
        monkeypatch.setattr(routing.sys, "platform", plat, raising=False)
        path = claude_desktop._config_path()
        assert path.name == expected_substring
        assert path.is_absolute()


def test_launch_runs_subprocess_with_ignore_certificate_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, cmd):
            calls.append(cmd)

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:  # pragma: no cover - unused in suite
            return None

    monkeypatch.setattr(routing.shutil, "which", lambda _: "/usr/bin/claude-desktop")
    monkeypatch.setattr(routing.subprocess, "Popen", FakeProcess)

    with pytest.raises(SystemExit) as exc:
        claude_desktop.launch(["/tmp/some extra path/"])
    assert exc.value.code == 0
    assert calls == [
        [
            "/usr/bin/claude-desktop",
            "--ignore-certificate-errors",
            "/tmp/some extra path/",
        ]
    ]


def test_launch_exits_127_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(routing.shutil, "which", lambda _: None)

    with pytest.raises(SystemExit) as exc:
        claude_desktop.launch([])
    assert exc.value.code == 127
    captured = capsys.readouterr()
    assert "claude-desktop" in captured.err


def test_launch_configure_does_not_invoke_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = tmp_path / "claude_desktop_config.json"
    invoked: list[bool] = []

    def fail_call(cmd):
        invoked.append(True)
        return 0

    monkeypatch.setattr(routing.subprocess, "Popen", fail_call)

    claude_desktop.launch(["--configure", "--config-path", str(fake)])

    assert invoked == []
    captured = capsys.readouterr()
    assert "Updated" in captured.out

    data = json.loads(fake.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
