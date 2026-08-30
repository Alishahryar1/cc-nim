"""Shared Claude Code environment policy for FCC client surfaces."""

from collections.abc import Mapping

from free_claude_code.cli.local_http import with_local_proxy_bypass

CLAUDE_CODE_AUTO_COMPACT_WINDOW = "190000"
CLAUDE_BINARY_NAME = "claude"

# Environment variable names FCC always sets itself; any inherited entry with
# one of these names is dropped before the block below re-sets it, regardless
# of casing.
_ANTHROPIC_PREFIX = "anthropic_"
_STRIPPED_KEYS = frozenset({"claude_code_disable_nonessential_traffic"})


def build_claude_proxy_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return the canonical environment for Claude Code proxy sessions."""

    # Windows environment variables are case-insensitive, so a leftover
    # differently-cased entry (e.g. `Anthropic_Base_Url`, set by an installer
    # or a Windows System Properties GUI) would survive a case-sensitive
    # startswith() filter and sit alongside the value we set below. The two
    # keys look distinct to this Python dict, but the child process's runtime
    # resolves env lookups case-insensitively on Windows, so which one wins
    # becomes ambiguous - in practice this is how Claude Code has been
    # observed silently falling back to the real Anthropic API instead of
    # the local proxy. Filtering by casefolded name avoids the ambiguity
    # entirely, on every platform.
    #
    # Claude's aggregate traffic flag also suppresses gateway model discovery.
    env = with_local_proxy_bypass(
        {
            key: value
            for key, value in base_env.items()
            if not key.casefold().startswith(_ANTHROPIC_PREFIX)
            and key.casefold() not in _STRIPPED_KEYS
        },
        proxy_root_url=proxy_root_url,
    )
    env["ANTHROPIC_BASE_URL"] = proxy_root_url
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = CLAUDE_CODE_AUTO_COMPACT_WINDOW
    env["DISABLE_AUTOUPDATER"] = "1"
    env["DISABLE_FEEDBACK_COMMAND"] = "1"
    env["DISABLE_ERROR_REPORTING"] = "1"
    return env
