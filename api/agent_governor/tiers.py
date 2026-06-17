"""Model size tiers and per-tier governor cap defaults.

Open-weight models span a huge size range (3B → 675B+). A single cap
schedule cannot serve a 30B nano *and* a 480B coder. This module assigns
each model id to a tier and supplies tier-appropriate caps.

Tier classification is **substring pattern matching** against the model id.
Patterns are tried from largest tier to smallest, first match wins. So a
model named ``mistral-large-3-675b-instruct-2512`` is classified as ``xxl``
because ``675b`` is matched before any smaller substring would be tried.

Frontier (Claude, GPT-5, Gemini, Grok) is matched first via the same
pattern lists used elsewhere in the governor — those models bypass entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelTier(StrEnum):
    """Coarse size class. Drives default caps and prompt augmentations."""

    FRONTIER = "frontier"  # Claude / GPT-5 / Gemini / Grok — bypass
    XXL = "xxl"  # 600B+: mistral-large-3-675b
    XL = "xl"  # 400-600B: qwen3-coder-480b, llama-3.1-405b, hermes-3-405b
    LARGE = "large"  # 200-400B: nemotron-ultra-253b, deepseek-v4-pro, qwen 235b
    MEDIUM = "medium"  # 80-200B: nemotron-super-120b, gpt-oss-120b, glm-4.6
    SMALL = "small"  # 30-80B: qwen3-next-80b, llama-70b, nemotron-49b, gpt-oss-20b
    TINY = "tiny"  # ≤30B: nemotron-nano-30b, llama-8b, gemma-9b
    UNKNOWN = "unknown"  # falls back to small defaults


@dataclass(slots=True, frozen=True)
class TierCaps:
    """Per-tier governor caps. Shipped as defaults, fully env-overridable."""

    max_consecutive_tool_calls: int
    max_total_tool_calls: int
    tool_cull_max: int
    plan_mode: bool
    termination_hint: bool


# Defaults reflect the practical balance between "free model can finish a
# real task" and "free model cannot loop overnight." Larger models get
# wider headroom because they're better at terminating on their own.
TIER_DEFAULTS: dict[ModelTier, TierCaps] = {
    ModelTier.TINY: TierCaps(
        max_consecutive_tool_calls=4,
        max_total_tool_calls=20,
        tool_cull_max=15,
        plan_mode=True,
        termination_hint=True,
    ),
    ModelTier.SMALL: TierCaps(
        max_consecutive_tool_calls=8,
        max_total_tool_calls=40,
        tool_cull_max=25,
        plan_mode=True,
        termination_hint=True,
    ),
    ModelTier.MEDIUM: TierCaps(
        max_consecutive_tool_calls=15,
        max_total_tool_calls=80,
        tool_cull_max=35,
        plan_mode=True,
        termination_hint=True,
    ),
    ModelTier.LARGE: TierCaps(
        max_consecutive_tool_calls=25,
        max_total_tool_calls=150,
        tool_cull_max=50,
        plan_mode=False,  # large coders plan well on their own
        termination_hint=True,
    ),
    ModelTier.XL: TierCaps(
        max_consecutive_tool_calls=80,
        max_total_tool_calls=400,
        tool_cull_max=80,
        plan_mode=False,
        termination_hint=True,
    ),
    ModelTier.XXL: TierCaps(
        max_consecutive_tool_calls=120,
        max_total_tool_calls=600,
        tool_cull_max=100,
        plan_mode=False,
        termination_hint=False,  # 675B+ generally terminate themselves
    ),
    ModelTier.UNKNOWN: TierCaps(
        max_consecutive_tool_calls=8,
        max_total_tool_calls=40,
        tool_cull_max=25,
        plan_mode=True,
        termination_hint=True,
    ),
}


# Substring patterns for each tier, **ordered largest tier → smallest tier**.
# First tier whose pattern list contains a substring of the (lowercased)
# model id wins. Frontier check happens elsewhere; this map is for non-
# frontier sizing only.
#
# Numeric size hints (``480b``, ``70b``) match the *total* parameter count
# in conventional model naming. MoE active-params suffixes like ``a35b``
# are intentionally NOT in the lists — we want total params, not active.
TIER_PATTERNS: tuple[tuple[ModelTier, tuple[str, ...]], ...] = (
    (
        ModelTier.XXL,
        (
            "675b",
            "qwen3-max",
            "mistral-large-3",
            "deepseek-r2",  # speculative; future-proof
        ),
    ),
    (
        ModelTier.XL,
        (
            "480b",
            "405b",
            "qwen3-coder-480b",
            "llama-3.1-405b",
            "hermes-3-llama-3.1-405b",
            "ring-2.6-1t",  # 1T-param MoE; treat as XL not XXL since active params are smaller
        ),
    ),
    (
        ModelTier.LARGE,
        (
            "253b",
            "235b",
            "236b",
            "230b",
            "200b",
            "ultra-253b",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "mistral-large-2",
        ),
    ),
    (
        ModelTier.MEDIUM,
        (
            "100b",
            "120b",
            "150b",
            "180b",
            "nemotron-super-120b",
            "gpt-oss-120b",
            "glm-4.6",
            "glm-4.5",
            "minimax-m2.5",
            "kimi-k2",
            "kimi-k2-instruct",
            "mistral-nemotron",  # ~115B
        ),
    ),
    (
        ModelTier.SMALL,
        (
            "32b",
            "40b",
            "49b",
            "60b",
            "65b",
            "70b",
            "72b",
            "78b",
            "80b",
            "qwen3-next-80b",
            "qwen3-coder-30b",
            "nemotron-3-super",  # safe even if numeric size missing
            "llama-3.3-70b",
            "llama-3.1-70b",
            "command-r-plus",
            "deepseek-coder-v2",
        ),
    ),
    (
        ModelTier.TINY,
        (
            "1b",
            "2b",
            "3b",
            "4b",
            "5b",
            "6b",
            "7b",
            "8b",
            "9b",
            "12b",
            "13b",
            "14b",
            "16b",
            "20b",
            "22b",
            "24b",
            "26b",
            "28b",
            "30b",
            "nano-30b",
            "nano-9b",
            "nano-12b",
            "lfm-2.5",
            "gemma-4-31b",
            "gemma-4-26b",
        ),
    ),
)


def detect_tier(model_id: str) -> ModelTier:
    """Return the size tier for a non-frontier model id.

    Frontier classification (claude-, gpt-5, gemini, grok) is handled by
    GovernorConfig.is_strong_model and bypasses this function. If you call
    it on a frontier id you'll get whatever size pattern matches first
    (usually MEDIUM via "100b"-ish substrings) — caller is responsible.
    """
    lowered = model_id.lower()
    for tier, patterns in TIER_PATTERNS:
        for pattern in patterns:
            if pattern in lowered:
                return tier
    return ModelTier.UNKNOWN


def caps_for_tier(tier: ModelTier) -> TierCaps:
    """Return the cap defaults for a given tier."""
    return TIER_DEFAULTS.get(tier, TIER_DEFAULTS[ModelTier.UNKNOWN])
