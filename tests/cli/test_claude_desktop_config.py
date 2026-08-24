"""Tests for the Claude Desktop config merge and binary launch helpers."""

import json
from pathlib import Path

import pytest

from free_claude_code.cli import claude_desktop, tls_proxy
from free_claude_code.config.settings import Settings


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    return tmp_path / "claude_desktop_config.json"


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(host="127.0.0.1", port=8082, proxy_auth_token="tok-123")


@pytest.fixture(autouse=True)
def deterministic_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin gateway URL resolution so tests never probe the network.

    The TLS-front probe is disabled at its source (a developer machine may
    have a live front on 8443); ``desktop_gateway_base_url`` still runs for
    real, so these tests also pin the desktop path prefix wiring.
    """

    monkeypatch.setattr(tls_proxy, "probe_https", lambda *a, **kw: False)


def test_managed_block_derives_url_and_key_from_settings(
    fake_settings: Settings,
) -> None:
    block = claude_desktop.fcc_managed_block(
        fake_settings, gateway_base_url="https://localhost:8443"
    )

    assert block["inferenceGatewayBaseUrl"] == "https://localhost:8443"
    assert block["inferenceAnthropicApiKey"] == "tok-123"
    assert block["provider"] == "gateway"


def test_managed_block_resolves_url_when_not_overridden(
    fake_settings: Settings,
) -> None:
    block = claude_desktop.fcc_managed_block(fake_settings)

    assert block["inferenceGatewayBaseUrl"] == ("http://127.0.0.1:8082/claude-desktop")


def test_configure_applies_block_when_file_is_missing(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    assert not fake_config.exists()

    changed = claude_desktop.configure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
    assert (
        data["inference"]["inferenceGatewayBaseUrl"]
        == "http://127.0.0.1:8082/claude-desktop"
    )
    assert data["inference"]["inferenceAnthropicApiKey"] == "tok-123"


def test_configure_is_idempotent(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    second = claude_desktop.configure_claude_desktop_config(
        fake_config, settings=fake_settings
    )
    assert second is False


def test_configure_preserves_unrelated_keys(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
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

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["preferences"] == {"theme": "dark"}
    assert data["mcpServers"] == {"fs": {"command": "node"}}
    assert data["coworkUserFilesPath"] == "/home/user/cowork"
    assert data["env"] == {"ANTHROPIC_BASE_URL": "https://localhost:8443"}
    assert data["modelDiscoveryEnabled"] is True


def test_configure_overwrites_partial_inference_block(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.write_text(
        json.dumps({"inference": {"provider": "anthropic", "keep_me": True}}),
        encoding="utf-8",
    )

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    inference = data["inference"]
    assert inference["provider"] == "gateway"
    assert inference["keep_me"] is True
    assert (
        inference["inferenceGatewayBaseUrl"] == "http://127.0.0.1:8082/claude-desktop"
    )


def test_configure_aborts_on_malformed_json(
    fake_config: Path,
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text("{not json", encoding="utf-8")

    changed = claude_desktop.configure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is False
    assert fake_config.read_text(encoding="utf-8") == "{not json"
    captured = capsys.readouterr()
    assert captured.out == ""


def test_configure_aborts_on_non_object_root(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text("[1, 2]", encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is False
    )


def test_load_existing_config_raises_malformed_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("nope", encoding="utf-8")

    with pytest.raises(claude_desktop.MalformedConfigError):
        claude_desktop.load_existing_config(path)


def test_unconfigure_drops_only_fcc_keys(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    managed = claude_desktop.fcc_managed_block(fake_settings)
    fake_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "modelDiscoveryEnabled": True,
                "inference": managed | {"extra_user_key": "stay"},
            }
        ),
        encoding="utf-8",
    )

    changed = claude_desktop.unconfigure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in data
    assert data["inference"] == {"extra_user_key": "stay"}
    assert data["preferences"] == {"theme": "dark"}


def test_unconfigure_roundtrip_is_clean_noop(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data == {}

    # Second passes are no-ops.
    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is False
    )


def test_unconfigure_silent_on_missing_file(tmp_path: Path) -> None:
    assert (
        claude_desktop.unconfigure_claude_desktop_config(tmp_path / "absent.json")
        is False
    )


def test_config_path_resolver_returns_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    for platform in ("linux", "darwin", "win32"):
        monkeypatch.setattr(claude_desktop.sys, "platform", platform)
        path = claude_desktop._config_path()
        assert path.name == claude_desktop.CONFIG_FILENAME
        assert path.is_absolute()


def test_main_configure_writes_block_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    fake = tmp_path / "claude_desktop_config.json"

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configure must not spawn a subprocess")

    monkeypatch.setattr(claude_desktop.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        claude_desktop, "get_settings", lambda: fake_settings, raising=False
    )

    claude_desktop.main(["--configure", "--config-path", str(fake)])

    data = json.loads(fake.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True


def test_launch_binary_spawns_with_certificate_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            calls.append(cmd)

    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", FakeProcess)

    claude_desktop.launch_binary(["--extra"])

    assert calls == [["/usr/bin/claude", "--ignore-certificate-errors", "--extra"]]


def test_launch_binary_raises_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: None)

    with pytest.raises(FileNotFoundError):
        claude_desktop.launch_binary()


def test_ensure_configured_and_launch_merges_then_spawns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    order: list[str] = []
    original_configure = claude_desktop.configure_claude_desktop_config

    def fake_configure(
        path: Path | None = None, settings: Settings | None = None
    ) -> bool:
        order.append("configure")
        return original_configure(path or config_path, settings=settings)

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            order.append(f"spawn:{cmd[0]}")

    monkeypatch.setattr(
        claude_desktop, "configure_claude_desktop_config", fake_configure
    )
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", FakeProcess)

    claude_desktop.ensure_configured_and_launch(settings=fake_settings)

    assert order[0] == "configure"
    assert order[1].startswith("spawn:")
