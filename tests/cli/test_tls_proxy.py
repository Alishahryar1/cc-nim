"""Tests for the HTTPS front proxy: probing, Caddyfile rendering, lifecycle."""

import email.message
import threading
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.cli import tls_proxy
from free_claude_code.cli.tls_proxy import CaddyTlsProxy, resolve_gateway_base_url
from free_claude_code.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(host="127.0.0.1", port=8082, tls_proxy_port=18443)


@pytest.fixture
def http_server() -> Iterator[HTTPServer]:
    """Local plain-HTTP server used to exercise the HTTPS probe negatively."""

    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server
    server.shutdown()
    server.server_close()


def test_probe_https_false_against_plain_http_listener(
    http_server: HTTPServer,
) -> None:
    # An HTTPS handshake against a plain-HTTP listener must fail the probe.
    assert (
        tls_proxy.probe_https(f"https://127.0.0.1:{http_server.server_port}") is False
    )


def test_probe_https_false_when_nothing_listens(settings: Settings) -> None:
    assert tls_proxy.probe_https(tls_proxy.tls_root_url(settings)) is False


def test_render_caddyfile_fronts_local_server(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-test-caddy"))

    text = proxy.render_caddyfile()

    assert "localhost:18443 {" in text
    assert "tls internal" in text
    assert "reverse_proxy 127.0.0.1:8082 {" in text
    assert "flush_interval -1" in text


def test_resolve_gateway_base_url_https_when_probe_succeeds(
    settings: Settings,
) -> None:
    with patch.object(tls_proxy, "probe_https", return_value=True):
        url = resolve_gateway_base_url(settings)

    assert url == "https://localhost:18443"


def test_resolve_gateway_base_url_http_when_disabled_or_unreachable(
    settings: Settings,
) -> None:
    disabled = Settings(host="127.0.0.1", port=8082, tls_proxy_enabled=False)

    assert resolve_gateway_base_url(disabled) == "http://127.0.0.1:8082"
    assert resolve_gateway_base_url(settings) == "http://127.0.0.1:8082"


def test_start_reuses_external_front_without_spawning(
    settings: Settings,
    tmp_path: Path,
) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy, "probe_https", side_effect=[True]),
        patch.object(tls_proxy.shutil, "which") as which,
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        started = proxy.start()

    assert started is True
    which.assert_not_called()
    popen.assert_not_called()


def test_start_spawns_managed_caddy_and_waits_for_readiness(
    settings: Settings,
    tmp_path: Path,
) -> None:
    caddyfile_dir = tmp_path / "caddy"

    class FakeProcess:
        def __init__(self, cmd: list[str], **_kwargs: object) -> None:
            self.cmd = cmd
            self.returncode = 0

        def poll(self) -> int | None:
            return None  # stays alive while readiness probes run

        def terminate(self) -> None: ...

        def wait(self, timeout: float | None = None) -> int:
            return 0

    probe_calls = 0

    def counting_probe(url: str, timeout_seconds: float = 1.0) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        # First call (external-front check) fails; readiness checks succeed.
        return probe_calls > 1

    with (
        patch.object(tls_proxy, "probe_https", counting_probe),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", FakeProcess),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=caddyfile_dir)
        started = proxy.start()

    assert started is True
    caddyfile = (caddyfile_dir / "Caddyfile").read_text(encoding="utf-8")
    assert "tls internal" in caddyfile


def test_start_falls_back_to_http_without_caddy_binary(
    settings: Settings,
    tmp_path: Path,
) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy, "probe_https", return_value=False),
        patch.object(tls_proxy.shutil, "which", return_value=None),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        started = proxy.start()

    assert started is False
    popen.assert_not_called()


def test_start_reports_child_that_exits_immediately(
    settings: Settings,
    tmp_path: Path,
) -> None:
    class DyingProcess:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.returncode = 1

        def poll(self) -> int:
            return self.returncode

    with (
        patch.object(tls_proxy, "probe_https", return_value=False),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", DyingProcess),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")
        started = proxy.start()

    assert started is False


def test_stop_terminates_owned_running_process(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-unused"))

    process = MagicMock()
    process.poll.return_value = None
    proxy._process = process

    proxy.stop()

    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_stop_is_noop_after_child_already_exited(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-unused"))

    process = MagicMock()
    process.poll.return_value = 0
    proxy._process = process

    proxy.stop()

    process.terminate.assert_not_called()


def test_managed_caddyfile_disables_admin_api(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-test-caddy"))

    # A system caddy may already own the default admin endpoint (port 2019);
    # the managed instance must never claim it.
    assert "admin off" in proxy.render_caddyfile()


@pytest.mark.parametrize("flag", ["--config", "--adapter"])
def test_spawn_command_includes_config_adapter(flag: str, settings: Settings) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            self.returncode = 0

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    with (
        # First probe (external-front check) fails; readiness probe succeeds.
        patch.object(tls_proxy, "probe_https", side_effect=[False, True]),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", FakeProcess),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-cmd-check"))
        proxy.start()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "/usr/bin/caddy" in cmd
    assert flag in cmd


def test_probe_https_true_when_endpoint_returns_error_status() -> None:
    # Any HTTP status (404 root path, 401 auth gate, 502 dead upstream)
    # proves the TLS front is alive and answering.
    error = urllib.error.HTTPError(
        "https://localhost:18443/",
        404,
        "Not Found",
        email.message.Message(),
        None,
    )

    with patch.object(
        tls_proxy.urllib.request, "urlopen", MagicMock(side_effect=error)
    ):
        assert tls_proxy.probe_https("https://localhost:18443/") is True
