"""Spawn-only managed HTTPS front for clients that refuse plain HTTP.

``CaddyTlsProxy`` owns one FCC-managed ``caddy run`` child process: a
generated Caddyfile under the sandboxed FCC config dir, ``tls internal``
self-signed certificate, admin API disabled, and all of Caddy's state
jailed inside that directory via XDG environment overrides. The front
reverse-proxies to the local FCC server.

Fail-fast semantics: ``start()`` raises ``FrontStartError`` instead of
falling back. Claude Desktop cannot route through plain HTTP, so a front
that cannot come up aborts the desktop host loudly. There is no adoption
of external listeners and no code path that can emit an ``http://``
gateway URL; clients trust the internal certificate via
``--ignore-certificate-errors``.
"""

import http.client
import os
import shutil
import ssl
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

from free_claude_code.config import paths
from free_claude_code.config.server_urls import _browser_host_for_local_urls
from free_claude_code.config.settings import Settings

CADDY_BINARY = "caddy"
CADDY_HOME_DIRNAME = "caddy"
READY_TIMEOUT_SECONDS = 10.0
READY_PROBE_TIMEOUT_SECONDS = 0.5
READY_STEP_SECONDS = 0.25


class FrontStartError(RuntimeError):
    """The managed HTTPS front could not be brought up."""


def tls_root_url(settings: Settings) -> str:
    """HTTPS root the desktop front serves on.

    Always ``localhost``: the front is local by definition, and the managed
    caddy requests its internal certificate for that name.
    """

    return f"https://localhost:{settings.tls_proxy_port}"


def desktop_gateway_base_url(settings: Settings, base_url: str | None = None) -> str:
    """Desktop-scoped gateway URL: TLS root plus the desktop path prefix.

    The prefix routes Claude Desktop onto the alias-emitting API mount while
    every other FCC client keeps hitting the bare paths. ``base_url``
    overrides the root for callers that already know which front serves.
    A plain-HTTP root is unreachable by construction: the desktop gateway
    only ever exists when a TLS front is up.
    """

    root = base_url if base_url is not None else tls_root_url(settings)
    prefix = settings.desktop_gateway_prefix.strip("/")
    return f"{root.rstrip('/')}/{prefix}"


def _front_healthy(target: str, timeout: float = 1.0) -> bool:
    """Whether the front answers ``/health`` with a 2xx over TLS.

    The certificate is ``tls internal`` self-signed for ``localhost``, so
    verification is deliberately skipped — we own the only listener this
    probe can reach.
    """

    request = urllib.request.Request(f"{target.rstrip('/')}/health")
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            code = response.getcode()
            return code is not None and 200 <= code < 300
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        http.client.HTTPException,
    ):
        return False


def _default_home_dir() -> Path:
    """Sandbox directory for the managed front's state and certificates.

    Routed through ``paths.config_dir_path()`` so the test-suite config-dir
    monkeypatch is honored; ``FCC_CONFIG_HOME`` stays a hard override.
    """

    override = os.environ.get("FCC_CONFIG_HOME")
    base = Path(override) if override else paths.config_dir_path()
    return base / CADDY_HOME_DIRNAME


def _restrict_home_permissions(directory: Path) -> None:
    """Tighten an existing Caddy home to owner-only access.

    ``mkdir(mode=0o700)`` only applies to directories the call itself
    creates; a home that already exists keeps whatever mode it was made
    with, so under a permissive umask an inherited ``0777`` home would let
    any local user traverse and write into the directory holding the
    front's Caddyfile and CA state. The directory is forced to owner-only
    before any of those files are written.
    """

    current = directory.stat().st_mode
    target = stat.S_IRWXU
    if current & 0o777 != target:
        directory.chmod(target)


class CaddyTlsProxy:
    """Lifecycle owner of one FCC-managed ``caddy run`` child process."""

    def __init__(self, settings: Settings, home_dir: Path | None = None) -> None:
        self._settings = settings
        self._home_dir = home_dir or _default_home_dir()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def caddyfile_path(self) -> Path:
        return self._home_dir / "Caddyfile"

    def render_caddyfile(self) -> str:
        """Caddyfile text fronting the local FCC server with an internal cert."""

        upstream_host = _browser_host_for_local_urls(self._settings)
        if ":" in upstream_host:
            upstream_host = f"[{upstream_host}]"
        upstream = f"{upstream_host}:{self._settings.port}"
        return "\n".join(
            [
                "{",
                "    admin off",
                # The automatic HTTP->HTTPS redirect would try to bind :80,
                # which unprivileged users cannot do. Disabling just the
                # redirect keeps automatic certificate management alive.
                "    auto_https disable_redirects",
                "}",
                f"localhost:{self._settings.tls_proxy_port} {{",
                "    tls internal",
                f"    reverse_proxy {upstream} {{",
                "        flush_interval -1",
                "    }",
                "}",
                "",
            ]
        )

    def _write_caddyfile(self, text: str) -> None:
        """Write the Caddyfile owner-only, never world-readable."""

        descriptor = os.open(
            self.caddyfile_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        self.caddyfile_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def start(self) -> None:
        """Spawn the managed front and wait until it answers; raise otherwise.

        Spawn-only: there is no probe or adoption of an existing listener and
        no plain-HTTP fallback. Every failure mode — missing binary, unusable
        state directory, spawn failure, early child exit (including a port
        already owned by another listener), or readiness timeout — stops any
        partial child and raises ``FrontStartError``.
        """

        if not self._settings.tls_proxy_enabled:
            return

        target = tls_root_url(self._settings)
        binary = shutil.which(CADDY_BINARY)
        if binary is None:
            self.stop()
            raise FrontStartError(
                "caddy binary not found on PATH — install caddy or set "
                "TLS_PROXY_ENABLED=false"
            )

        try:
            self._home_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            _restrict_home_permissions(self._home_dir)
            self._write_caddyfile(self.render_caddyfile())
        except OSError as exc:
            self.stop()
            raise FrontStartError(
                f"Could not prepare the managed caddy state directory "
                f"{self._home_dir}: {exc}"
            ) from exc

        env = dict(os.environ)
        env["XDG_DATA_HOME"] = str(self._home_dir / "data")
        env["XDG_CONFIG_HOME"] = str(self._home_dir / "config")
        try:
            self._process = subprocess.Popen(
                [
                    binary,
                    "run",
                    "--config",
                    str(self.caddyfile_path),
                    "--adapter",
                    "caddyfile",
                ],
                cwd=self._home_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self._process = None
            self.stop()
            raise FrontStartError(f"Could not start managed caddy: {exc}") from exc

        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                exit_code = self._process.returncode
                self._process = None
                self.stop()
                raise FrontStartError(
                    f"The managed HTTPS front could not start on {target}: "
                    f"another listener owns the port, or caddy failed. "
                    f"Stop the other listener or change DESKTOP_TLS_PORT. "
                    f"(caddy exited with code {exit_code})"
                )
            if _front_healthy(target, timeout=READY_PROBE_TIMEOUT_SECONDS):
                logger.info("Managed HTTPS front ready at {}", target)
                return
            time.sleep(READY_STEP_SECONDS)

        self.stop()
        raise FrontStartError(
            f"The managed HTTPS front did not become ready on {target} within "
            f"{READY_TIMEOUT_SECONDS:.0f}s: another listener owns the port, "
            f"or caddy failed. Stop the other listener or change "
            f"DESKTOP_TLS_PORT."
        )

    def stop(self) -> None:
        """Terminate the managed child process, if we own one."""

        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
