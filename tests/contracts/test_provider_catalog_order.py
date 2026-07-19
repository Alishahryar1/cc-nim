"""Freeze ``PROVIDER_CATALOG`` insertion order used as canonical provider ranking."""

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
)

_EXPECTED_PROVIDER_ORDER: tuple[str, ...] = (
    "nvidia_nim",
    "open_router",
    "gemini",
    "vertex",
    "deepseek",
    "mistral",
    "mistral_codestral",
    "opencode",
    "opencode_go",
    "vercel",
    "bedrock",
    "huggingface",
    "cohere",
    "github_models",
    "wafer",
    "kimi",
    "kimi_code",
    "minimax",
    "cerebras",
    "groq",
    "sambanova",
    "fireworks",
    "cloudflare",
    "zai",
    "ollama_cloud",
    "lmstudio",
    "llamacpp",
    "ollama",
)


def test_provider_catalog_key_order_matches_canonical_plan() -> None:
    """NIM first; OpenCode pair stays adjacent; gateways precede native remotes."""

    assert tuple(PROVIDER_CATALOG.keys()) == _EXPECTED_PROVIDER_ORDER
    assert SUPPORTED_PROVIDER_IDS == _EXPECTED_PROVIDER_ORDER
