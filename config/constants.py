"""Shared defaults used by config models and provider adapters."""

# HTTP client connect timeout (seconds). Keep aligned with README.md and .env.example.
HTTP_CONNECT_TIMEOUT_DEFAULT = 10.0

# Anthropic Messages API default when the client omits max_tokens.
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 81920

# Z.ai GLM-5.2 max output tokens (128K). z.ai's Anthropic Messages endpoint
# accepts up to 131072 output tokens for glm-5.2; used when a request omits
# max_tokens. Scoped to Z.ai so other Anthropic-compat providers keep the shared
# default above.
ZAI_DEFAULT_MAX_OUTPUT_TOKENS = 131072

# Default Z.ai reasoning effort. GLM-5.x selects reasoning depth via a discrete
# top-level ``reasoning_effort`` field (``high`` | ``max``); ``max`` is z.ai's
# recommendation for coding. Override with ``ZAI_REASONING_EFFORT``
# (e.g. ``high`` or empty to disable).
ZAI_DEFAULT_REASONING_EFFORT = "max"
# Values of ``ZAI_REASONING_EFFORT`` that disable effort injection.
ZAI_REASONING_EFFORT_DISABLED_TOKENS = frozenset(
    {"", "0", "false", "no", "off", "none", "disabled"}
)
ZAI_REASONING_EFFORT_LEVELS = frozenset({"high", "max"})

# Claude Code auto-compaction window (tokens). The context size Claude Code uses
# to decide when to summarize the conversation. The default keeps a safe ceiling
# for most providers; raise to 1000000 (via ``CLAUDE_CODE_AUTO_COMPACT_WINDOW``)
# to utilize glm-5.2[1m]'s 1M context window.
CLAUDE_CODE_AUTO_COMPACT_WINDOW_DEFAULT = 190000

# Max bytes read from a non-200 native messages response when verbose error logging is on.
NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES = 4096
