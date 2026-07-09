"""Installed `fcc-claude` launcher."""

import os
import sys
from collections.abc import Mapping, Sequence

import httpx

from free_claude_code.api.admin_urls import local_proxy_root_url
from free_claude_code.cli.claude_env import (
    CLAUDE_BINARY_NAME,
    CLAUDE_CODE_AUTO_COMPACT_WINDOW,
    claude_auth_token,
)
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.settings import Settings, get_settings

from .common import preflight_proxy, resolve_client_binary, run_client_process

_DISPLAY_NAME = "Claude Code"
_INSTALL_HINT = "Install Claude Code with: npm install -g @anthropic-ai/claude-code"


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Claude Code with Free Claude Code proxy environment variables."""

    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    binary_name = claude_binary_name()
    binary_path = resolve_client_binary(
        binary_name=binary_name,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    args = list(sys.argv[1:] if argv is None else argv)
    run_client_process(
        command=build_claude_launcher_command(binary_path=binary_path, argv=args),
        env=build_claude_launcher_env(
            proxy_root_url=proxy_root_url,
            auth_token=settings.anthropic_auth_token,
            base_env=os.environ,
            auto_compact_window=resolve_auto_compact_window(settings),
        ),
        binary_name=binary_name,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )


def claude_binary_name() -> str:
    """Return the Claude Code binary name."""

    return CLAUDE_BINARY_NAME


def build_claude_launcher_command(
    *, binary_path: str, argv: Sequence[str]
) -> list[str]:
    """Return the Claude wrapper command without changing user arguments."""

    return [binary_path, *argv]


# Session-identity vars injected by a live Claude Code session. When fcc-claude
# is invoked from inside one (e.g. an agent's Bash tool), the nested claude.exe
# inherits these, decides it is a child session with host-managed OAuth, and
# ignores ANTHROPIC_AUTH_TOKEN entirely — every request then 401s against the
# proxy. fcc-claude's contract is a fresh standalone session pointed at the
# local proxy, so all inherited session/harness vars must be dropped; the two
# CLAUDE_CODE_* vars the launcher itself needs are re-set explicitly below.
_SESSION_ENV_VARS = frozenset(
    {
        "CLAUDECODE",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_EFFORT",
        "AI_AGENT",
        "BAGGAGE",
    }
)


def _is_inherited_session_var(key: str) -> bool:
    return (
        key in _SESSION_ENV_VARS
        or key.startswith("ANTHROPIC_")
        or key.startswith("CLAUDE_CODE_")
    )


def resolve_auto_compact_window(settings: Settings) -> str:
    """Match Claude Code's auto-compact window to the loaded local model.

    The static default (190000) is far above a locally-loaded context like
    45k/130k, so Claude Code never compacts before the local ceiling and LM
    Studio silently truncates the stream. When LM Studio is a configured
    provider, read the actually-loaded context length and reserve headroom for
    one large tool-result turn plus generation.

    Checks every configured model slot (MODEL and the MODEL_OPUS/SONNET/HAIKU
    overrides), not just MODEL: routing sends a Claude request through whichever
    override matches its tier, so an LM Studio override under a non-LM-Studio
    default must still get a truthful window. The compact window is a single
    global env var with no per-tier control, so if *any* slot is LM Studio we
    size to its ceiling — missing compaction there causes a silent truncation,
    whereas the same window applied to a large-context cloud model only compacts
    slightly early, which is harmless.
    """
    if not any(
        ref.provider_id == "lmstudio" for ref in configured_chat_model_refs(settings)
    ):
        return CLAUDE_CODE_AUTO_COMPACT_WINDOW
    try:
        root = settings.lm_studio_base_url.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        response = httpx.get(f"{root}/api/v0/models", timeout=2.0)
        response.raise_for_status()
        loaded = [
            model.get("loaded_context_length")
            for model in response.json().get("data", [])
            if model.get("state") == "loaded"
            and isinstance(model.get("loaded_context_length"), int)
        ]
        if not loaded:
            return CLAUDE_CODE_AUTO_COMPACT_WINDOW
        # Use the SMALLEST loaded context, not the largest: the compact window
        # is a single global env var applied to every request, so if several
        # models are loaded at different contexts (e.g. a small haiku-tier model
        # and a large opus-tier one), a request routed to the smallest one must
        # still compact before it truncates. Sizing to max() would leave the
        # small model unprotected — the exact silent truncation this prevents.
        context_length = min(loaded)
        window = max(8192, min(int(context_length * 0.85), context_length - 12000))
        # The 8192 floor above is meant to avoid compacting too early on modest
        # contexts, not to override a model loaded with an even smaller context
        # than that. Clamp to context_length so the window can never exceed the
        # actual ceiling — otherwise Claude Code still wouldn't compact before
        # LM Studio silently truncates the stream.
        window = min(window, context_length)
        return str(window)
    except Exception:
        return CLAUDE_CODE_AUTO_COMPACT_WINDOW


def build_claude_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
    auto_compact_window: str | None = None,
) -> dict[str, str]:
    """Return a Claude Code environment that targets the local proxy."""

    env = {
        key: value
        for key, value in base_env.items()
        if not _is_inherited_session_var(key)
    }
    env["ANTHROPIC_BASE_URL"] = proxy_root_url
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = (
        auto_compact_window or CLAUDE_CODE_AUTO_COMPACT_WINDOW
    )
    env["ANTHROPIC_AUTH_TOKEN"] = claude_auth_token(auth_token)
    return env
