"""Installed ``fcc-claude-desktop`` console script.

Thin launcher shim: every config-merge, unmerge, and binary-spawn behavior
lives in :mod:`free_claude_code.cli.claude_desktop` so the tray shell, the
desktop startup hook, and this entry point share one implementation. This
module only preserves the console-script surface used by the install and
uninstall scripts.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.cli import claude_desktop as routing
from free_claude_code.cli.tls_proxy import (
    ensure_https_front,
    verified_https_gateway_url,
)
from free_claude_code.config.loader import get_settings

CLAUDE_DESKTOP_BINARY = routing.CLAUDE_DESKTOP_BINARY

configure_claude_desktop_config = routing.configure_claude_desktop_config
unconfigure_claude_desktop_config = routing.unconfigure_claude_desktop_config
_config_path = routing._config_path
load_existing_config = routing.load_existing_config
MalformedConfigError = routing.MalformedConfigError


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fcc-claude-desktop",
        description=(
            "Launch Claude Desktop with the Free Claude Code routing layer, "
            "or merge/reverse the FCC routing block in claude_desktop_config.json."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--configure",
        action="store_true",
        help="Merge the FCC 3P-gateway block into Claude Desktop's config.",
    )
    mode.add_argument(
        "--unconfigure",
        action="store_true",
        help="Reverse the merge applied by --configure.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Override the Claude Desktop config path (primarily for tests).",
    )
    parser.add_argument(
        "binary_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the claude-desktop binary.",
    )
    return parser


def launch(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``fcc-claude-desktop`` console script."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = _build_argparser().parse_args(args)

    if parsed.configure:
        # --configure persists the gateway credential, so it only writes
        # against a verified HTTPS front: without one the URL would fall
        # back to plain HTTP and persist the reusable proxy token against
        # a cleartext endpoint any local process could occupy. Adoption is
        # probe-only — this command exits right after writing, so a front
        # it spawned itself would die with it and leave the config it just
        # wrote pointing at a dead gateway. The front must already be up
        # (FCC desktop host or the managed caddy proxy) and stay up.
        gateway_url = verified_https_gateway_url(get_settings())
        if gateway_url is None:
            print(
                "Refusing to configure Claude Desktop: no verified FCC "
                "HTTPS front is available. Enable the caddy TLS proxy (or "
                "start the FCC desktop host, which manages it) and retry.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        changed = configure_claude_desktop_config(
            parsed.config_path, gateway_base_url=gateway_url
        )
        routing.report_configure_result(parsed.config_path or _config_path(), changed)
        return

    if parsed.unconfigure:
        changed = unconfigure_claude_desktop_config(parsed.config_path)
        target = parsed.config_path or _config_path()
        print(f"{'Removed FCC block' if changed else 'Nothing to remove'}: {target}")
        return

    # Claude Desktop cannot route through plain HTTP, so bring up a
    # verified FCC HTTPS front before writing the routing block or
    # spawning the binary. Without one the gateway URL would fall back to
    # an unusable http:// address, so refuse to launch instead.
    settings = get_settings()
    tls_front = ensure_https_front(settings)
    if tls_front is None:
        print(
            "Refusing to launch Claude Desktop: no verified FCC HTTPS front "
            "is available. Enable the caddy TLS proxy (or start the FCC "
            "desktop host, which manages it) and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        gateway_url = verified_https_gateway_url(settings)
        if gateway_url is None:
            print(
                "Refusing to launch Claude Desktop: the FCC HTTPS front "
                "stopped verifying before the routing block could be "
                "written. Retry once the front is healthy.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        try:
            configure_claude_desktop_config(
                parsed.config_path, gateway_base_url=gateway_url
            )
        except OSError as exc:
            print(
                f"Warning: could not merge the FCC routing block: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from None

        # A malformed config cannot carry FCC routing; launching anyway would
        # start Claude Desktop without model discovery or inference routing.
        target = parsed.config_path or _config_path()
        try:
            load_existing_config(target)
        except FileNotFoundError:
            pass  # configure created a fresh config; nothing to validate.
        except MalformedConfigError as exc:
            print(
                f"Refusing to launch Claude Desktop: malformed config at "
                f"{exc.path}: {exc.reason}. Fix or remove the file and retry.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None

        try:
            process = routing.launch_binary(tuple(parsed.binary_args))
        except FileNotFoundError:
            print(
                f"Could not find '{CLAUDE_DESKTOP_BINARY}' binary.",
                file=sys.stderr,
            )
            print(
                "Install Claude Desktop from https://claude.ai/download.",
                file=sys.stderr,
            )
            raise SystemExit(127) from None
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        raise SystemExit(return_code)
    finally:
        tls_front.stop()
