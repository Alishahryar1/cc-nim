"""Tests for the HTTPS front proxy: probing, Caddyfile rendering, lifecycle."""

import email.message
import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.cli import tls_proxy
from free_claude_code.cli.tls_proxy import CaddyTlsProxy, resolve_gateway_base_url
from free_claude_code.config.settings import Settings

TEST_IDENTITY_SECRET = "fcc-test-identity-secret"


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


def test_probe_fcc_front_false_against_plain_http_listener(
    http_server: HTTPServer,
    tmp_path: Path,
) -> None:
    # An HTTPS handshake against a plain-HTTP listener must fail the probe.
    home = tmp_path / "caddy"
    with patch.object(tls_proxy, "managed_ca_path", return_value=Path("ca.crt")):
        assert (
            tls_proxy.probe_fcc_front(
                f"https://127.0.0.1:{http_server.server_port}",
                TEST_IDENTITY_SECRET,
                home,
            )
            is False
        )


def test_probe_fcc_front_false_when_nothing_listens(
    settings: Settings,
    tmp_path: Path,
) -> None:
    home = tmp_path / "caddy"
    with patch.object(tls_proxy, "managed_ca_path", return_value=Path("ca.crt")):
        assert (
            tls_proxy.probe_fcc_front(
                tls_proxy.tls_root_url(settings), TEST_IDENTITY_SECRET, home
            )
            is False
        )


def test_render_caddyfile_fronts_local_server(settings: Settings) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-test-caddy"))

    text = proxy.render_caddyfile(TEST_IDENTITY_SECRET)

    assert "localhost:18443 {" in text
    assert "tls internal" in text
    assert "reverse_proxy 127.0.0.1:8082 {" in text
    assert "flush_interval -1" in text


def test_render_caddyfile_serves_identity_secret_at_private_path(
    settings: Settings,
) -> None:
    # The managed front must answer the identity probe itself so later
    # launches can prove ownership before adopting it.
    proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-test-caddy"))

    text = proxy.render_caddyfile(TEST_IDENTITY_SECRET)

    assert f'respond /.fcc-front-identity "{TEST_IDENTITY_SECRET}" 200' in text


def _health_response(payload: bytes | str) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = (
        payload.encode() if isinstance(payload, str) else payload
    )
    return response


def test_probe_fcc_front_true_when_listener_serves_identity_secret(
    tmp_path: Path,
) -> None:
    home = tmp_path / "caddy"
    ca_file = tmp_path / "ca.crt"
    ca_file.write_text("placeholder")
    with (
        patch.object(tls_proxy, "managed_ca_path", return_value=ca_file),
        patch.object(tls_proxy, "_front_ssl_context") as context,
        patch.object(
            tls_proxy.urllib.request,
            "urlopen",
            MagicMock(return_value=_health_response(TEST_IDENTITY_SECRET)),
        ) as urlopen,
    ):
        assert (
            tls_proxy.probe_fcc_front(
                "https://localhost:18443", TEST_IDENTITY_SECRET, home
            )
            is True
        )

    request = urlopen.call_args.args[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "https://localhost:18443/.fcc-front-identity"
    # The request must run through the CA-pinned context, never an
    # unverified one.
    assert urlopen.call_args.kwargs["context"] is context.return_value
    context.assert_called_once_with(ca_file)


def test_probe_fcc_front_rejects_forged_health_marker(tmp_path: Path) -> None:
    # Regression guard for the Greptile "unauthenticated health marker
    # permits local TLS-front impersonation" finding: the publicly forgeable
    # /health payload must never count as front identity, even though it is
    # exactly what FCC's own /health endpoint serves.
    home = tmp_path / "caddy"
    ca_file = tmp_path / "ca.crt"
    ca_file.write_text("placeholder")
    with (
        patch.object(tls_proxy, "managed_ca_path", return_value=ca_file),
        patch.object(tls_proxy, "_front_ssl_context"),
        patch.object(
            tls_proxy.urllib.request,
            "urlopen",
            MagicMock(return_value=_health_response('{"status": "healthy"}')),
        ),
    ):
        assert (
            tls_proxy.probe_fcc_front(
                "https://localhost:18443", TEST_IDENTITY_SECRET, home
            )
            is False
        )


def test_probe_fcc_front_false_for_foreign_or_broken_listener(
    tmp_path: Path,
) -> None:
    # An unrelated local listener that answers anything other than this
    # install's identity secret must never be adopted as the gateway.
    home = tmp_path / "caddy"
    ca_file = tmp_path / "ca.crt"
    ca_file.write_text("placeholder")
    error = urllib.error.HTTPError(
        "https://localhost:18443/.fcc-front-identity",
        404,
        "Not Found",
        email.message.Message(),
        None,
    )
    responses = [
        _health_response(b"a-different-secret"),  # wrong secret
        _health_response(b'{"status": "healthy"}'),  # forged health marker
        _health_response(b""),  # empty body
        _health_response(b"\xff\xfe"),  # non-UTF-8 body
    ]
    errors = [
        error,  # non-200 status through the TLS layer
        urllib.error.URLError("connection refused"),  # nothing listening
        ssl.SSLCertVerificationError("certificate verify failed"),  # foreign cert
    ]
    with (
        patch.object(tls_proxy, "managed_ca_path", return_value=ca_file),
        patch.object(tls_proxy, "_front_ssl_context"),
    ):
        for response in responses:
            with patch.object(
                tls_proxy.urllib.request,
                "urlopen",
                MagicMock(return_value=response),
            ):
                assert (
                    tls_proxy.probe_fcc_front(
                        "https://localhost:18443/", TEST_IDENTITY_SECRET, home
                    )
                    is False
                )
        for error_side_effect in errors:
            with patch.object(
                tls_proxy.urllib.request,
                "urlopen",
                MagicMock(side_effect=error_side_effect),
            ):
                assert (
                    tls_proxy.probe_fcc_front(
                        "https://localhost:18443/", TEST_IDENTITY_SECRET, home
                    )
                    is False
                )


def test_probe_fcc_front_false_without_identity_secret(tmp_path: Path) -> None:
    # No secret means nothing to compare against; never adopt.
    assert tls_proxy.probe_fcc_front("https://localhost:18443", "", tmp_path) is False


def test_probe_fcc_front_false_without_managed_ca_file(tmp_path: Path) -> None:
    # Regression guard for the Greptile "reusable front identity plus
    # disabled TLS verification" finding: without this install's managed CA
    # root there is nothing to verify the listener's certificate against, so
    # even a listener serving the exact secret must not be adopted.
    home = tmp_path / "caddy"
    tls_proxy.front_identity_path(home).parent.mkdir(parents=True)
    assert tls_proxy.managed_ca_path(home).is_file() is False

    with patch.object(
        tls_proxy.urllib.request,
        "urlopen",
        MagicMock(return_value=_health_response(TEST_IDENTITY_SECRET)),
    ) as urlopen:
        assert (
            tls_proxy.probe_fcc_front(
                "https://localhost:18443", TEST_IDENTITY_SECRET, home
            )
            is False
        )

    urlopen.assert_not_called()  # no request ever leaves over unverified TLS


def _self_signed_cert(directory: Path, common_name: str) -> tuple[Path, Path]:
    key = directory / f"{common_name}.key"
    cert = directory / f"{common_name}.crt"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            f"/CN={common_name}",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    return key, cert


def test_front_ssl_context_rejects_foreign_certificate(tmp_path: Path) -> None:
    # The pinned context must refuse a certificate that does not chain to
    # the managed CA root — the exact shape of a same-user process replaying
    # the identity secret behind its own self-signed listener.
    if shutil.which("openssl") is None:
        pytest.skip("openssl binary not installed")

    _, ca_file = _self_signed_cert(tmp_path, "managed-ca")
    foreign_key, foreign_cert = _self_signed_cert(tmp_path, "foreign")

    context = tls_proxy._front_ssl_context(ca_file)
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True

    server_socket = socket.socket()
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(1)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(foreign_cert, foreign_key)

    def accept_once() -> None:
        connection, _ = server_socket.accept()
        try:
            server_context.wrap_socket(connection, server_side=True)
        except ssl.SSLError:
            pass  # expected: the client rejects our certificate mid-handshake
        finally:
            connection.close()

    worker = threading.Thread(target=accept_once, daemon=True)
    worker.start()
    try:
        port = server_socket.getsockname()[1]
        with (
            socket.create_connection(("127.0.0.1", port), timeout=5) as raw,
            pytest.raises(ssl.SSLCertVerificationError),
        ):
            context.wrap_socket(raw, server_hostname="localhost")
    finally:
        server_socket.close()


def test_managed_ca_path_tracks_sandboxed_caddy_data(
    settings: Settings,
    tmp_path: Path,
) -> None:
    home = tmp_path / "caddy"

    assert tls_proxy.managed_ca_path(home) == (
        home / "data" / "caddy" / "pki" / "authorities" / "local" / "root.crt"
    )
    # The default home must match what CaddyTlsProxy sandboxes via
    # XDG_DATA_HOME, otherwise the probe could never verify the child's
    # certificate.
    proxy = CaddyTlsProxy(settings)
    assert tls_proxy.managed_ca_path(None) == (
        proxy._home_dir
        / "data"
        / "caddy"
        / "pki"
        / "authorities"
        / "local"
        / "root.crt"
    )


def test_load_or_create_front_identity_is_owner_only_and_stable(
    tmp_path: Path,
) -> None:
    home = tmp_path / "caddy"

    first = tls_proxy.load_or_create_front_identity(home)
    second = tls_proxy.load_or_create_front_identity(home)

    assert first == second
    assert len(first) == tls_proxy.FRONT_IDENTITY_BYTES * 2  # hex-encoded
    secret_file = tls_proxy.front_identity_path(home)
    assert secret_file.stat().st_mode & 0o777 == 0o600
    assert secret_file.parent.stat().st_mode & 0o777 == 0o700


def test_load_or_create_front_identity_survives_permissive_umask(
    tmp_path: Path,
) -> None:
    # Under umask 000 the secret must still end up owner-only.
    previous = os.umask(0)
    try:
        tls_proxy.load_or_create_front_identity(tmp_path / "caddy")
    finally:
        os.umask(previous)

    secret_file = tls_proxy.front_identity_path(tmp_path / "caddy")
    assert secret_file.stat().st_mode & 0o777 == 0o600


def test_resolve_gateway_base_url_https_when_probe_succeeds(
    settings: Settings,
) -> None:
    with (
        patch.object(tls_proxy, "load_or_create_front_identity", return_value="secret"),
        patch.object(tls_proxy, "probe_fcc_front", return_value=True) as probe,
    ):
        url = resolve_gateway_base_url(settings)

    assert url == "https://localhost:18443"
    probe.assert_called_once_with("https://localhost:18443", "secret")


def test_resolve_gateway_base_url_https_requires_fcc_identity(
    settings: Settings,
) -> None:
    # A TLS listener that is alive but cannot prove it belongs to this FCC
    # install must not be adopted; routing falls back to plain HTTP.
    with (
        patch.object(tls_proxy, "load_or_create_front_identity", return_value="s"),
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
    ):
        url = resolve_gateway_base_url(settings)

    assert url == "http://127.0.0.1:8082"


def test_resolve_gateway_base_url_http_when_disabled_or_unreachable(
    settings: Settings,
) -> None:
    disabled = Settings(host="127.0.0.1", port=8082, tls_proxy_enabled=False)

    assert resolve_gateway_base_url(disabled) == "http://127.0.0.1:8082"
    # The enabled-but-unverified case must not touch the real home dir or
    # network: pin identity and force the probe to fail.
    with (
        patch.object(tls_proxy, "load_or_create_front_identity", return_value="s"),
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
    ):
        assert resolve_gateway_base_url(settings) == "http://127.0.0.1:8082"


def test_ensure_https_front_returns_proxy_when_start_succeeds(
    settings: Settings,
) -> None:
    proxy = MagicMock(spec=CaddyTlsProxy)
    proxy.start.return_value = True

    with patch.object(tls_proxy, "CaddyTlsProxy", return_value=proxy):
        front = tls_proxy.ensure_https_front(settings)

    assert front is proxy
    proxy.stop.assert_not_called()


def test_ensure_https_front_returns_none_and_stops_when_start_fails(
    settings: Settings,
) -> None:
    # A failed bring-up must leave no managed child behind: the gate stops
    # the proxy it constructed and hands the caller ``None``.
    proxy = MagicMock(spec=CaddyTlsProxy)
    proxy.start.return_value = False

    with patch.object(tls_proxy, "CaddyTlsProxy", return_value=proxy):
        front = tls_proxy.ensure_https_front(settings)

    assert front is None
    proxy.stop.assert_called_once()


def test_verified_https_gateway_url_returns_url_when_probe_passes(
    settings: Settings,
) -> None:
    with (
        patch.object(tls_proxy, "load_or_create_front_identity", return_value="secret"),
        patch.object(tls_proxy, "probe_fcc_front", return_value=True) as probe,
    ):
        url = tls_proxy.verified_https_gateway_url(settings)

    assert url == f"https://localhost:18443/{settings.desktop_gateway_prefix}"
    probe.assert_called_once_with("https://localhost:18443", "secret")


def test_verified_https_gateway_url_never_falls_back_to_http(
    settings: Settings,
) -> None:
    # Unlike resolve_gateway_base_url, the desktop gate must return None —
    # never a plain-HTTP URL — when the front does not verify.
    disabled = Settings(host="127.0.0.1", port=8082, tls_proxy_enabled=False)
    assert tls_proxy.verified_https_gateway_url(disabled) is None

    with (
        patch.object(tls_proxy, "load_or_create_front_identity", return_value="s"),
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
    ):
        assert tls_proxy.verified_https_gateway_url(settings) is None


def test_start_reuses_external_front_without_spawning(
    settings: Settings,
    tmp_path: Path,
) -> None:
    proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")

    with (
        patch.object(tls_proxy, "probe_fcc_front", side_effect=[True]) as probe,
        patch.object(tls_proxy.shutil, "which") as which,
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        started = proxy.start()

    assert started is True
    # Adoption must be checked against this install's managed CA.
    assert probe.call_args.args[2] == tmp_path / "caddy"
    which.assert_not_called()
    popen.assert_not_called()


def test_start_does_not_adopt_foreign_tls_listener(
    settings: Settings,
    tmp_path: Path,
) -> None:
    # A TLS endpoint that is alive on the port but does not front this FCC
    # server must not be adopted; the managed Caddy path runs instead.

    class FakeProcess:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.returncode = 0

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None: ...

        def wait(self, timeout: float | None = None) -> int:
            return 0

    probe_calls = 0

    def counting_probe(
        url: str,
        secret: str,
        home_dir: Path | None = None,
        timeout_seconds: float = 1.0,
    ) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return False  # identity fails; readiness probes also fail

    with (
        patch.object(tls_proxy, "probe_fcc_front", counting_probe),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", FakeProcess),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=tmp_path / "caddy")
        started = proxy.start()

    assert started is False
    assert probe_calls > 0  # readiness probes ran against our own child


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

    def counting_probe(
        url: str,
        secret: str,
        home_dir: Path | None = None,
        timeout_seconds: float = 1.0,
    ) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        # First call (external-front check) fails; readiness checks succeed.
        return probe_calls > 1

    with (
        patch.object(tls_proxy, "probe_fcc_front", counting_probe),
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
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
        patch.object(tls_proxy.shutil, "which", return_value=None),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        started = proxy.start()

    assert started is False
    popen.assert_not_called()


def test_start_swallows_state_preparation_errors(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """Filesystem failures while preparing the caddy home fall back to HTTP.

    ``start()`` promises never to raise: ``ServerSupervisor._run_once``
    calls it outside any try/except of its own, so an escaping ``OSError``
    from the home-directory or Caddyfile write would kill the generation
    before the plain-HTTP fallback could serve.
    """

    unwritable = tmp_path / "occupied"
    unwritable.write_text("not a directory", encoding="utf-8")

    with (
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        proxy = CaddyTlsProxy(settings, home_dir=unwritable / "caddy")
        started = proxy.start()

    assert started is False
    popen.assert_not_called()


def test_start_swallows_front_identity_errors(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """Filesystem failures while creating the identity secret stay swallowed.

    ``load_or_create_front_identity`` runs before the caddy state write
    and needs the same never-raise guarantee: without a secret there is
    nothing to prove front ownership with, so the front is not adoptable
    and the plain-HTTP fallback must apply instead of an escaping
    ``OSError`` killing the serving generation.
    """

    unwritable = tmp_path / "occupied"
    unwritable.write_text("not a directory", encoding="utf-8")

    with (
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen") as popen,
    ):
        proxy = CaddyTlsProxy(settings, home_dir=unwritable / "caddy")
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
        patch.object(tls_proxy, "probe_fcc_front", return_value=False),
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
    assert "admin off" in proxy.render_caddyfile(TEST_IDENTITY_SECRET)


def test_start_writes_caddyfile_owner_only(
    settings: Settings,
    tmp_path: Path,
) -> None:
    # The Caddyfile embeds the identity secret; it must not be world-readable.
    class FakeProcess:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.returncode = 0

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None: ...

        def wait(self, timeout: float | None = None) -> int:
            return 0

    probe_calls = 0

    def ready_on_second_probe(
        url: str,
        secret: str,
        home_dir: Path | None = None,
        timeout_seconds: float = 1.0,
    ) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return probe_calls > 1

    home = tmp_path / "caddy"
    with (
        patch.object(tls_proxy, "probe_fcc_front", ready_on_second_probe),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", FakeProcess),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=home)
        assert proxy.start() is True

    assert proxy.caddyfile_path.stat().st_mode & 0o777 == 0o600


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port_free(port: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with socket.socket() as probe_sock:
            try:
                probe_sock.connect(("127.0.0.1", port))
            except OSError:
                return
        time.sleep(0.1)


def test_managed_front_serves_identity_and_rejects_impersonation(
    tmp_path: Path,
) -> None:
    """End-to-end: the managed front proves ownership; impostors cannot.

    Spawns the real managed caddy (skipped when absent) and verifies the
    adoption probe accepts the front only with this install's secret —
    the exact path Greptile's T-Rex exploited with a forged health marker.
    """

    if shutil.which(tls_proxy.CADDY_BINARY) is None:
        pytest.skip("caddy binary not installed")

    port = _free_port()
    front_settings = Settings(host="127.0.0.1", port=1, tls_proxy_port=port)
    home = tmp_path / "caddy"
    proxy = CaddyTlsProxy(front_settings, home_dir=home)
    base_url = f"https://localhost:{port}"
    try:
        assert proxy.start() is True
        secret = tls_proxy.load_or_create_front_identity(home)

        assert tls_proxy.probe_fcc_front(base_url, secret, home) is True
        # A process that merely occupies the port cannot guess the secret.
        assert tls_proxy.probe_fcc_front(base_url, "attacker-guess", home) is False
    finally:
        proxy.stop()
        _wait_for_port_free(port)


def test_managed_front_rejects_attacker_certificate_replaying_secret(
    tmp_path: Path,
) -> None:
    """End-to-end guard for the Greptile reusable-bearer finding.

    A same-user process can read the identity secret from disk and serve it
    from its own listener. Adoption must still fail because that listener's
    self-signed certificate does not chain to this install's managed CA.
    """

    if shutil.which(tls_proxy.CADDY_BINARY) is None or shutil.which("openssl") is None:
        pytest.skip("caddy or openssl binary not installed")

    port = _free_port()
    front_settings = Settings(host="127.0.0.1", port=1, tls_proxy_port=port)
    home = tmp_path / "caddy"
    proxy = CaddyTlsProxy(front_settings, home_dir=home)
    base_url = f"https://localhost:{port}"
    try:
        assert proxy.start() is True
        secret = tls_proxy.load_or_create_front_identity(home)
    finally:
        proxy.stop()
        _wait_for_port_free(port)

    # Attacker: same user, so it can read the secret; it binds the port with
    # its own self-signed certificate and replays the secret verbatim.
    attacker_key, attacker_cert = _self_signed_cert(tmp_path, "attacker")

    class IdentityHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = secret.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", port), IdentityHandler)
    attacker_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    attacker_context.load_cert_chain(attacker_cert, attacker_key)
    server.socket = attacker_context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        # The secret matches byte-for-byte; only the certificate differs.
        assert tls_proxy.probe_fcc_front(base_url, secret, home) is False
    finally:
        server.shutdown()
        server.server_close()
        _wait_for_port_free(port)


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

    probe_calls = 0

    def ready_on_second_probe(
        url: str,
        secret: str,
        home_dir: Path | None = None,
        timeout_seconds: float = 1.0,
    ) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return probe_calls > 1

    with (
        patch.object(tls_proxy, "probe_fcc_front", ready_on_second_probe),
        patch.object(tls_proxy.shutil, "which", return_value="/usr/bin/caddy"),
        patch.object(tls_proxy.subprocess, "Popen", FakeProcess),
    ):
        proxy = CaddyTlsProxy(settings, home_dir=Path("/tmp/fcc-cmd-check"))
        proxy.start()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "/usr/bin/caddy" in cmd
    assert flag in cmd
