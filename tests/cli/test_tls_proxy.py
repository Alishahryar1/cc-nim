"""Tests for the spawn-only managed HTTPS front proxy."""

import email.message
import http.client
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.cli import tls_proxy
from free_claude_code.cli.tls_proxy import (
    CaddyTlsProxy,
    FrontStartError,
    desktop_gateway_base_url,
    tls_root_url,
)
from free_claude_code.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(host="127.0.0.1", port=8082, tls_proxy_port=18443)


# ==================== URL helpers ====================


def test_tls_root_url_uses_local_host_and_configured_port(
    settings: Settings,
) -> None:
    assert tls_root_url(settings) == "https://localhost:18443"


def test_desktop_gateway_base_url_joins_prefix_onto_tls_root(
    settings: Settings,
) -> None:
    assert (
        desktop_gateway_base_url(settings) == "https://localhost:18443/claude-desktop"
    )


def test_desktop_gateway_base_url_accepts_explicit_root_override(
    settings: Settings,
) -> None:
    url = desktop_gateway_base_url(settings, base_url="https://example.test:9443/")

    assert url == "https://example.test:9443/claude-desktop"


def test_desktop_gateway_base_url_never_emits_plain_http(settings: Settings) -> None:
    # The desktop mount is credential-bearing: it is only ever served through
    # the TLS front, so no input to the join can produce an http:// URL.
    url = desktop_gateway_base_url(settings, base_url=tls_root_url(settings))

    assert url.startswith("https://")


# ==================== Caddyfile rendering ====================


def test_render_caddyfile_fronts_local_server(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-test-caddy"))

    text = proxy.render_caddyfile()

    assert "localhost:18443 {" in text
    assert "tls internal" in text
    assert "reverse_proxy 127.0.0.1:8082 {" in text
    assert "flush_interval -1" in text


def test_managed_caddyfile_disables_admin_api(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-test-caddy"))

    # A system caddy may already own the default admin endpoint (port 2019);
    # the managed instance must never claim it.
    assert "admin off" in proxy.render_caddyfile()


def test_render_caddyfile_disables_http_redirect_binding(settings: Settings) -> None:
    # The automatic HTTP->HTTPS redirect would try to bind :80, which
    # unprivileged users cannot do.
    assert (
        "auto_https disable_redirects"
        in CaddyTlsProxy(
            settings, home_dir=Path("/tmp/fcc-test-caddy")
        ).render_caddyfile()
    )


# ==================== readiness helper ====================


def test_front_healthy_true_for_2xx_response() -> None:
    response = MagicMock()
    response.__enter__.return_value = response
    response.getcode.return_value = 200

    with patch.object(
        tls_proxy.urllib.request, "urlopen", MagicMock(return_value=response)
    ) as urlopen:
        assert tls_proxy._front_healthy("https://localhost:18443/") is True

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://localhost:18443/health"


@pytest.mark.parametrize("status", [404, 502])
def test_front_healthy_false_for_non_2xx_status(status: int) -> None:
    headers = email.message.Message()
    error = urllib.error.HTTPError(
        "https://localhost:18443/health",
        status,
        "Not Found",
        headers,
        None,
    )

    with patch.object(
        tls_proxy.urllib.request, "urlopen", MagicMock(side_effect=error)
    ):
        assert tls_proxy._front_healthy("https://localhost:18443") is False
    # An HTTPError carries a temporary body file; close it so its GC cannot
    # fire a ResourceWarning inside a later test (warnings are errors here).
    error.close()


def test_front_healthy_false_on_transport_errors() -> None:
    errors: list[Exception] = [
        urllib.error.URLError("connection refused"),
        OSError("broken pipe"),
        ValueError("bad url"),
        http.client.HTTPException("handshake failed"),
    ]
    for error in errors:
        with patch.object(
            tls_proxy.urllib.request, "urlopen", MagicMock(side_effect=error)
        ):
            assert tls_proxy._front_healthy("https://localhost:18443") is False


# ==================== start(): fail-fast semantics ====================


class _AliveProcess:
    """Fake caddy child that stays alive while readiness probes run."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.returncode = 0

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int:
        return 0


class _CapturingProcess(_AliveProcess):
    """Fake child that records the spawn command and keyword arguments."""

    cmd: list[str]
    kwargs: dict[str, object]

    def __init__(self, cmd: list[str], **kwargs: object) -> None:
        super().__init__(cmd, **kwargs)
        self.cmd = cmd
        self.kwargs = kwargs


def _ready_after_first_probe() -> Callable[[str, float], bool]:
    """Readiness probe that fails once (caddy booting) then succeeds."""
    calls = 0

    def probe(target: str, timeout: float = 1.0) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    return probe


def test_start_raises_front_start_error_without_caddy_binary(
    settings: Settings,
    tmp_path: Path,
) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy.shutil, "which", return_value=None),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
        pytest.raises(FrontStartError, match="install caddy"),
    ):
        proxy.start()

    popen.assert_not_called()
    proxy.stop()  # fail-fast path already cleaned up; stop stays a no-op


def test_start_raises_front_start_error_when_child_exits_immediately(
    settings: Settings,
    tmp_path: Path,
) -> None:
    class DyingProcess:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.returncode = 1

        def poll(self) -> int:
            return self.returncode

        def terminate(self) -> None: ...

        def wait(self, timeout: float | None = None) -> int:
            return 1

    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", DyingProcess),
        pytest.raises(FrontStartError, match="another listener owns the port"),
    ):
        proxy.start()


def test_start_raises_front_start_error_on_state_preparation_failure(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """Filesystem failures while preparing the caddy home fail fast.

    ``start()`` raises ``FrontStartError`` instead of swallowing the OSError:
    the desktop host has no plain-HTTP fallback, so a half-prepared front must
    abort startup loudly (this reverses the earlier swallow-and-continue
    behavior that existed to support an HTTP fallback which no longer exists).
    """

    unwritable = tmp_path / "occupied"
    unwritable.write_text("not a directory", encoding="utf-8")

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
        pytest.raises(FrontStartError),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=unwritable / "caddy")
        proxy.start()

    popen.assert_not_called()


def test_start_raises_front_start_error_on_spawn_oserror(
    settings: Settings,
    tmp_path: Path,
) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", side_effect=OSError("no exec")),
        pytest.raises(FrontStartError, match="no exec"),
    ):
        proxy.start()


def test_start_raises_front_start_error_on_readiness_timeout(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Never-healthy child: the wall-clock deadline must expire and fail fast.
    monkeypatch.setattr(tls_proxy, "READY_TIMEOUT_SECONDS", 0.5)

    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", _AliveProcess),
        patch.object(tls_proxy, "_front_healthy", return_value=False) as healthy,
        pytest.raises(FrontStartError, match="did not become ready"),
    ):
        proxy.start()

    assert healthy.call_count >= 1
    assert proxy._process is None  # timeout path stopped the child


def test_start_is_noop_when_tls_disabled(settings: Settings, tmp_path: Path) -> None:
    # Disabled front must not spawn anything and must not raise: the host
    # simply skips the TLS front and prints no gateway URL.
    disabled = Settings.model_construct(
        host="127.0.0.1", port=8082, tls_proxy_enabled=False, tls_proxy_port=18443
    )
    disabled_proxy = CaddyTlsProxy(disabled, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        disabled_proxy.start()  # no raise, no spawn

    popen.assert_not_called()
    assert disabled_proxy._process is None


# ==================== start(): success path ====================


def test_start_spawns_managed_caddy_and_waits_for_readiness(
    settings: Settings,
    tmp_path: Path,
) -> None:
    caddyfile_dir = tmp_path / "caddy"
    probe = _ready_after_first_probe()

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", _AliveProcess),
        patch.object(tls_proxy, "_front_healthy", probe),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=caddyfile_dir)
        proxy.start()  # returns None on success

    caddyfile = (caddyfile_dir / "Caddyfile").read_text(encoding="utf-8")
    assert "tls internal" in caddyfile
    assert proxy._process is not None


@pytest.mark.parametrize("flag", ["--config", "--adapter"])
def test_spawn_command_includes_config_adapter(
    flag: str, settings: Settings, tmp_path: Path
) -> None:
    spawned: list[_CapturingProcess] = []

    class RecordingProcess(_CapturingProcess):
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            super().__init__(cmd, **kwargs)
            spawned.append(self)

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", RecordingProcess),
        patch.object(tls_proxy, "_front_healthy", _ready_after_first_probe()),
    ):
        CaddyTlsProxy(settings, home_dir=tmp_path / "caddy").start()

    assert len(spawned) == 1
    assert "/usr/bin/caddy" in spawned[0].cmd
    assert flag in spawned[0].cmd


def test_start_jails_child_env_and_working_directory(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """Caddy state stays inside the sandboxed home via the XDG env jail."""

    spawned: list[_CapturingProcess] = []
    home = tmp_path / "caddy"

    class RecordingProcess(_CapturingProcess):
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            super().__init__(cmd, **kwargs)
            spawned.append(self)

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", RecordingProcess),
        patch.object(tls_proxy, "_front_healthy", _ready_after_first_probe()),
    ):
        CaddyTlsProxy(settings, home_dir=home).start()

    assert len(spawned) == 1
    assert spawned[0].kwargs["cwd"] == home
    env = spawned[0].kwargs["env"]
    assert isinstance(env, dict)
    assert env["XDG_DATA_HOME"] == str(home / "data")
    assert env["XDG_CONFIG_HOME"] == str(home / "config")


def test_start_detaches_child_into_new_session(
    settings: Settings,
    tmp_path: Path,
) -> None:
    # start_new_session keeps the managed caddy out of the host's process
    # group so a Ctrl-C aimed at the host cannot orphan-or-kill it midway.
    spawned: list[_CapturingProcess] = []

    class RecordingProcess(_CapturingProcess):
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            super().__init__(cmd, **kwargs)
            spawned.append(self)

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", RecordingProcess),
        patch.object(tls_proxy, "_front_healthy", _ready_after_first_probe()),
    ):
        CaddyTlsProxy(settings, home_dir=tmp_path / "caddy").start()

    assert len(spawned) == 1
    assert spawned[0].kwargs["start_new_session"] is True


# ==================== sandbox permissions ====================


def test_start_tightens_existing_permissive_home(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """The managed front's state directory is forced owner-only on start.

    ``mkdir(mode=0o700)`` only governs directories the call itself creates,
    so a home that already exists with 0777 (inherited under a permissive
    umask or created by an older install) would keep letting other local
    users traverse and write into the directory that holds the front's
    Caddyfile and CA state. Creation must restrict the existing directory
    itself. ``chmod`` (not mkdir's mode arg) defeats the test process's
    own umask, which would otherwise leave the directory at 0775.
    """

    home = tmp_path / "caddy"
    home.mkdir()
    home.chmod(0o777)
    assert home.stat().st_mode & 0o777 == 0o777

    proxy = tls_proxy.CaddyTlsProxy(settings, home_dir=home)

    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", _AliveProcess),
        patch.object(tls_proxy, "_front_healthy", return_value=True),
    ):
        proxy.start()

    assert home.stat().st_mode & 0o777 == 0o700


def test_start_writes_caddyfile_owner_only(
    settings: Settings,
    tmp_path: Path,
) -> None:
    probe_calls = 0

    def ready_on_second_probe(target: str, timeout: float = 1.0) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return probe_calls > 1

    home = tmp_path / "caddy"
    with (
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", _AliveProcess),
        patch.object(tls_proxy, "_front_healthy", ready_on_second_probe),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=home)
        proxy.start()

    assert proxy.caddyfile_path.stat().st_mode & 0o777 == 0o600


# ==================== stop() ====================


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


def test_stop_kills_child_when_terminate_times_out(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-unused"))

    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired("caddy", 5),  # terminate wait times out
        0,  # kill wait succeeds
    ]
    proxy._process = process

    proxy.stop()

    process.kill.assert_called_once()
    process.wait.assert_called()


def test_stop_is_noop_when_nothing_owned(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-unused"))

    proxy.stop()  # must not raise


# ==================== default home dir ====================


def test_default_home_dir_lives_under_config_dir(tmp_path: Path) -> None:
    from free_claude_code.config import paths

    with patch.object(paths, "config_dir_path", return_value=tmp_path):
        home = tls_proxy._default_home_dir()

    assert home == tmp_path / "caddy"


def test_default_home_dir_respects_fcc_config_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FCC_CONFIG_HOME", str(tmp_path / "override"))

    assert tls_proxy._default_home_dir() == tmp_path / "override" / "caddy"


# ==================== real-caddy smoke test (single) ====================


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _HealthHandler(BaseHTTPRequestHandler):
    """Tiny upstream standing in for the FCC server's /health."""

    def do_GET(self) -> None:
        # BaseHTTPRequestHandler API name; overridden as required by the stdlib.
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(
        self, format: str, *args: Any
    ) -> None: ...  # silence request logging; name matches the stdlib base


@pytest.mark.live
@pytest.mark.skipif(shutil.which("caddy") is None, reason="caddy not installed")
def test_managed_front_end_to_end_with_real_caddy(tmp_path: Path) -> None:
    """Live smoke: spawn caddy, probe /health through TLS, stop, verify cleanup.

    The only test allowed to spawn a real caddy. An upstream answering /health
    on the FCC port stands in for the server (the host starts the front before
    the server; readiness through the front requires an answering upstream).
    """

    upstream_port = _free_port()
    tls_port = _free_port()

    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), _HealthHandler)
    upstream_worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_worker.start()
    try:
        live_settings = Settings(
            host="127.0.0.1",
            port=upstream_port,
            tls_proxy_port=tls_port,
        )
        home = tmp_path / "caddy"
        proxy = CaddyTlsProxy(live_settings, home_dir=home)
        try:
            proxy.start()
            target = tls_root_url(live_settings)
            assert tls_proxy._front_healthy(target, timeout=2.0) is True
        finally:
            proxy.stop()

        # The managed child must be gone: no caddy serving our Caddyfile.
        listing = subprocess.run(
            ["pgrep", "-af", "caddy run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert str(proxy.caddyfile_path) not in (listing.stdout or "")
        assert proxy._process is None

        # The TLS port must stop answering once the front is stopped.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket() as probe_sock:
                try:
                    probe_sock.connect(("127.0.0.1", tls_port))
                except OSError:
                    break
            time.sleep(0.1)
    finally:
        upstream.shutdown()
        upstream.server_close()
