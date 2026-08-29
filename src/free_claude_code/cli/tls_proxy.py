"""HTTPS front proxy for clients that refuse plain HTTP (e.g. Claude Desktop).

Three cooperating pieces:

1. ``probe_fcc_front`` / ``resolve_gateway_base_url`` — detect whether an
   HTTPS endpoint that fronts *this* FCC install already answers on the
   configured TLS port. An adopted front also receives clients' API keys, so
   adoption demands two independent proofs: the listener's certificate must
   chain to this install's managed CA root (ownership of the key material an
   unrelated local process cannot forge), and it must serve the per-install
   identity secret at a private path. Any verified front is reused as-is;
   FCC never touches it.
2. ``CaddyTlsProxy`` — when nothing answers and a ``caddy`` binary exists,
   spawn an FCC-managed instance: generated Caddyfile under ``~/.fcc/caddy/``
   with ``tls internal``, admin API disabled, and all of Caddy's state
   sandboxed inside that directory. The proxy forwards to the local FCC
   server and is stopped alongside it.
3. ``ensure_https_front`` / ``verified_https_gateway_url`` — the gate
   credential-bearing entry points (the standalone launcher) use: bring up a
   verified HTTPS front or fail, and only then hand out the gateway URL.
   Claude Desktop cannot use the plain-HTTP fallback, so these never return
   one.

The long-running server (``commands.ServerSupervisor``) additionally runs a
concurrent readiness thread that re-probes with ``probe_fcc_front`` while
Uvicorn serves — the front can only verify once FCC's own routes answer
through it — and upgrades the published desktop gateway URL to the
TLS-prefixed one during the live serving window.

Security boundary: every proof here (the identity secret, the managed CA
key material) is owner-only on disk, which keeps *other local users* out.
A process already running as this user can read all of them, so same-user
code execution is outside the security boundary this module defends — the
same boundary the proxy auth token itself lives behind.
"""

import http.client
import os
import secrets
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
from free_claude_code.config.server_urls import (
    _browser_host_for_local_urls,
    local_proxy_root_url,
)
from free_claude_code.config.settings import Settings

CADDY_BINARY = "caddy"
CADDY_HOME_DIRNAME = "caddy"
READY_TIMEOUT_SECONDS = 10.0
# Private path the managed front answers on to prove it belongs to this FCC
# install. Unrelated to any route FCC serves, so a reverse proxy in front of
# FCC cannot collide with or forge it.
FRONT_IDENTITY_PATH = "/.fcc-front-identity"
FRONT_IDENTITY_FILENAME = "front-identity"
FRONT_IDENTITY_BYTES = 32
# Where the managed caddy stores its internal CA, relative to the sandboxed
# XDG_DATA_HOME set in ``CaddyTlsProxy.start``.
_MANAGED_CA_PARTS = ("data", "caddy", "pki", "authorities", "local", "root.crt")


def _default_home_dir() -> Path:
    """Sandbox directory for the managed front's state and identity proofs."""

    override = os.environ.get("FCC_CONFIG_HOME")
    base = Path(override) if override else paths.config_dir_path()
    return base / CADDY_HOME_DIRNAME


def front_identity_path(home_dir: Path | None = None) -> Path:
    """File holding this install's front-identity secret (owner-only)."""

    base = home_dir or _default_home_dir()
    return base / FRONT_IDENTITY_FILENAME


def managed_ca_path(home_dir: Path | None = None) -> Path:
    """Root certificate of this install's managed caddy internal CA.

    The managed caddy sandboxes all state under ``home_dir`` (its
    ``XDG_DATA_HOME``), so its CA root lives at a fixed path inside it.
    """

    base = home_dir or front_identity_path().parent
    return base.joinpath(*_MANAGED_CA_PARTS)


def _front_ssl_context(ca_path: Path) -> ssl.SSLContext:
    """TLS context that only trusts certificates issued by ``ca_path``.

    Full chain validation plus hostname verification (``urllib`` pins the
    request URL's host as the expected certificate name): a local process
    that merely binds the port presents a self-signed certificate that does
    not chain to this install's managed CA, so the handshake fails before
    any request is sent to it.
    """

    context = ssl.create_default_context(cafile=str(ca_path))
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.hostname_checks_common_name = False
    return context


def load_or_create_front_identity(home_dir: Path | None = None) -> str:
    """Return this install's front-identity secret, creating it once.

    The secret is one of the two proofs a TLS listener belongs to this FCC
    install before it is adopted as the credential-bearing gateway (the
    other is its certificate chaining to the managed CA), so it must be
    unguessable (256 random bits) and readable only by the owner regardless
    of the process umask.
    """

    path = front_identity_path(home_dir)
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret = secrets.token_hex(FRONT_IDENTITY_BYTES)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secret)
    # ``os.open`` mode is still filtered by umask at creation; force
    # owner-only so a permissive umask cannot widen it.
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return secret


def probe_fcc_front(
    base_url: str,
    identity_secret: str,
    home_dir: Path | None = None,
    timeout_seconds: float = 1.0,
) -> bool:
    """Whether ``base_url`` fronts *this* FCC install, not just any TLS listener.

    Adopting an external front is a trust decision: whatever answers also
    receives clients' API keys and inference traffic. A listener therefore
    only counts as an existing front when both hold:

    * its certificate validates against this install's managed CA root with
      hostname verification — a reusable secret alone is a bearer value any
      same-user process could read from disk and replay over its own
      self-signed TLS, but such a listener's certificate never chains to the
      managed CA;
    * it serves this install's unguessable identity secret at the private
      identity path.

    Without the managed CA file there is nothing to verify against, so the
    listener is never adopted and callers fall back to plain HTTP.
    """

    if not identity_secret:
        return False
    ca_path = managed_ca_path(home_dir)
    if not ca_path.is_file():
        return False
    try:
        context = _front_ssl_context(ca_path)
    except OSError:
        return False
    identity_url = f"{base_url.rstrip('/')}{FRONT_IDENTITY_PATH}"
    request = urllib.request.Request(identity_url)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds, context=context
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        return False
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        http.client.HTTPException,
    ):
        return False
    try:
        served = body.decode("utf-8").strip()
    except UnicodeDecodeError:
        return False
    return secrets.compare_digest(served, identity_secret)


def tls_root_url(settings: Settings) -> str:
    """HTTPS URL Claude Desktop should use instead of the plain HTTP root.

    Always ``localhost``: the TLS front is local by definition, and the
    managed Caddy instance requests its internal certificate for that name.
    """

    return f"https://localhost:{settings.tls_proxy_port}"


def resolve_gateway_base_url(settings: Settings) -> str:
    """Best gateway base URL: HTTPS when this install's front owns the TLS port.

    Only a listener that proves it belongs to this FCC install (certificate
    chained to the managed CA plus identity secret match) is advertised as
    the gateway; anything else falls back to plain HTTP so credentials are
    never routed to an unverified listener.
    """

    if settings.tls_proxy_enabled and probe_fcc_front(
        tls_root_url(settings), load_or_create_front_identity()
    ):
        return tls_root_url(settings)
    return local_proxy_root_url(settings)


def desktop_gateway_base_url(settings: Settings, base_url: str | None = None) -> str:
    """Gateway URL scoped to Claude Desktop: root plus the desktop path prefix.

    The prefix routes Claude Desktop onto the alias-emitting API mount while
    every other FCC client keeps hitting the bare paths, which always serve
    raw provider refs. ``base_url`` overrides resolution for callers that
    already know which root serves (e.g. a front the readiness thread just
    verified).
    """

    base = resolve_gateway_base_url(settings) if base_url is None else base_url
    return f"{base.rstrip('/')}/{settings.desktop_gateway_prefix}"


def ensure_https_front(settings: Settings) -> CaddyTlsProxy | None:
    """Bring up a verified HTTPS front, or return ``None`` when impossible.

    Reuses an already-verified front without spawning anything; otherwise
    starts the managed proxy and keeps it running only while it stays
    verified. Callers that receive ``None`` must not write a Desktop routing
    block or spawn Claude Desktop: without HTTPS the gateway URL would fall
    back to plain HTTP, which Claude Desktop cannot use.
    """

    proxy = CaddyTlsProxy(settings)
    if proxy.start():
        return proxy
    proxy.stop()
    return None


def verified_https_gateway_url(settings: Settings) -> str | None:
    """Desktop gateway URL, but only when a verified HTTPS front answers.

    Returns ``None`` instead of the plain-HTTP fallback
    ``resolve_gateway_base_url`` would produce, because Claude Desktop
    cannot route through plain HTTP and a credential-bearing config must
    never point at an unverified listener.
    """

    if settings.tls_proxy_enabled and probe_fcc_front(
        tls_root_url(settings), load_or_create_front_identity()
    ):
        return f"{tls_root_url(settings).rstrip('/')}/{settings.desktop_gateway_prefix}"
    return None


class CaddyTlsProxy:
    """Lifecycle owner of one FCC-managed ``caddy run`` child process."""

    def __init__(self, settings: Settings, home_dir: Path | None = None) -> None:
        self._settings = settings
        self._home_dir = home_dir or _default_home_dir()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def caddyfile_path(self) -> Path:
        return self._home_dir / "Caddyfile"

    def render_caddyfile(self, identity_secret: str) -> str:
        """Caddyfile text fronting the local FCC server with an internal cert.

        The managed front also serves this install's identity secret at the
        private identity path so later launches can prove ownership before
        adopting it as the credential-bearing gateway.
        """

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
                f'    respond {FRONT_IDENTITY_PATH} "{identity_secret}" 200',
                f"    reverse_proxy {upstream} {{",
                "        flush_interval -1",
                "    }",
                "}",
                "",
            ]
        )

    def _write_caddyfile(self, text: str) -> None:
        """Write the Caddyfile owner-only: it embeds the identity secret."""

        descriptor = os.open(
            self.caddyfile_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        self.caddyfile_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def start(self) -> bool:
        """Start the managed proxy; returns whether HTTPS is now reachable.

        Reuses an existing external front — only after its certificate
        validates against this install's managed CA and it serves the
        identity secret — without spawning anything. Never raises: failures
        are logged and reported as ``False`` so callers can fall back to
        plain HTTP.
        """

        if not self._settings.tls_proxy_enabled:
            return False
        target = tls_root_url(self._settings)
        identity = load_or_create_front_identity(self._home_dir)
        if probe_fcc_front(target, identity, self._home_dir):
            logger.info("Reusing existing FCC HTTPS front at {}", target)
            return True

        binary = shutil.which(CADDY_BINARY)
        if binary is None:
            logger.warning(
                "No HTTPS front answered on {} and no caddy binary found; "
                "Claude Desktop routing stays on plain HTTP.",
                target,
            )
            return False

        try:
            self._home_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._write_caddyfile(self.render_caddyfile(identity))
        except OSError as exc:
            logger.warning(
                "Could not prepare the managed caddy state directory: {}; "
                "Claude Desktop routing stays on plain HTTP.",
                exc,
            )
            return False
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
            logger.warning("Could not start managed caddy: {}", exc)
            self._process = None
            return False

        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        step = 0.25
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                logger.warning(
                    "Managed caddy exited early with code {}; "
                    "Claude Desktop routing stays on plain HTTP.",
                    self._process.returncode,
                )
                self._process = None
                return False
            # Readiness means the full adoption probe passes against our own
            # child: certificate chained to the managed CA and identity
            # secret served. A transport-only handshake would also accept an
            # unrelated listener that raced us onto the port. The probe is
            # skipped until the child has written its CA root, so the
            # timeout must cover that bootstrap too.
            if probe_fcc_front(target, identity, self._home_dir, 0.5):
                logger.info("Managed HTTPS front ready at {}", target)
                return True
            time.sleep(step)

        logger.warning(
            "Managed caddy did not become ready on {} in {}s; "
            "Claude Desktop routing stays on plain HTTP.",
            target,
            READY_TIMEOUT_SECONDS,
        )
        self.stop()
        return False

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
