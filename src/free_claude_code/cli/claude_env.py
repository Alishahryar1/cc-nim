"""Shared Claude Code environment policy for FCC client surfaces."""

from collections.abc import Mapping

CLAUDE_CODE_AUTO_COMPACT_WINDOW = "190000"
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
CLAUDE_BINARY_NAME = "claude"
CLAUDE_NO_AUTH_SENTINEL = "fcc-no-auth"


def build_claude_proxy_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return the canonical environment for Claude Code proxy sessions."""

    env = {
        key: value
        for key, value in base_env.items()
        if not key.startswith("ANTHROPIC_")
    }
    env["ANTHROPIC_BASE_URL"] = proxy_root_url
    env["ANTHROPIC_AUTH_TOKEN"] = claude_auth_token(auth_token)
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = CLAUDE_CODE_AUTO_COMPACT_WINDOW
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = (
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC
    )
    return env


def claude_auth_token(auth_token: str) -> str:
    """Return the Claude Code auth marker for proxy-auth or no-auth sessions."""

    return auth_token.strip() or CLAUDE_NO_AUTH_SENTINEL
