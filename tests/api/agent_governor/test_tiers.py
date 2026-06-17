"""Tests for size-tier classification + per-tier cap defaults."""

from __future__ import annotations

import pytest

from api.agent_governor.config import (
    GovernorConfig,
    ModelOverride,
    load_governor_config,
)
from api.agent_governor.tiers import (
    TIER_DEFAULTS,
    ModelTier,
    caps_for_tier,
    detect_tier,
)


@pytest.mark.parametrize(
    "model_id, expected",
    [
        # Frontier → not handled by detect_tier; covered separately below.
        ("nvidia/llama-3.3-nemotron-super-49b-v1.5", ModelTier.SMALL),
        ("qwen/qwen3-next-80b-a3b-instruct", ModelTier.SMALL),
        ("meta-llama/llama-3.3-70b-instruct", ModelTier.SMALL),
        ("openai/gpt-oss-20b", ModelTier.TINY),
        ("nvidia/nemotron-3-nano-30b-a3b", ModelTier.TINY),
        ("nvidia/nemotron-3-super-120b-a12b", ModelTier.MEDIUM),
        ("openai/gpt-oss-120b", ModelTier.MEDIUM),
        ("z-ai/glm-4.5-air", ModelTier.MEDIUM),
        ("z-ai/glm-4.6", ModelTier.MEDIUM),
        ("nvidia/llama-3.1-nemotron-ultra-253b-v1", ModelTier.LARGE),
        ("deepseek-ai/deepseek-v4-pro", ModelTier.LARGE),
        ("qwen/qwen3-coder-480b-a35b-instruct", ModelTier.XL),
        ("nousresearch/hermes-3-llama-3.1-405b", ModelTier.XL),
        ("inclusionai/ring-2.6-1t", ModelTier.XL),
        ("mistralai/mistral-large-3-675b-instruct-2512", ModelTier.XXL),
    ],
)
def test_detect_tier_known_models(model_id: str, expected: ModelTier):
    assert detect_tier(model_id) == expected


def test_detect_tier_unknown_returns_unknown():
    assert detect_tier("totally/made-up-model") == ModelTier.UNKNOWN


def test_caps_for_tier_returns_defaults():
    caps = caps_for_tier(ModelTier.XL)
    assert caps == TIER_DEFAULTS[ModelTier.XL]
    assert caps.max_consecutive_tool_calls == 80


def test_tier_caps_strictly_increase_with_size():
    """Bigger tier → wider headroom across all numeric caps."""
    ordered = (
        ModelTier.TINY,
        ModelTier.SMALL,
        ModelTier.MEDIUM,
        ModelTier.LARGE,
        ModelTier.XL,
        ModelTier.XXL,
    )
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        l_caps, r_caps = TIER_DEFAULTS[left], TIER_DEFAULTS[right]
        assert l_caps.max_consecutive_tool_calls < r_caps.max_consecutive_tool_calls
        assert l_caps.max_total_tool_calls < r_caps.max_total_tool_calls
        assert l_caps.tool_cull_max < r_caps.tool_cull_max


def test_governor_config_tier_for_strong_returns_frontier():
    cfg = GovernorConfig()
    assert cfg.tier_for("claude-sonnet-4-6") == ModelTier.FRONTIER
    assert cfg.tier_for("gpt-5") == ModelTier.FRONTIER


def test_governor_config_for_model_uses_tier_defaults():
    """A vanilla config falls back to tier defaults when env caps are unset."""
    cfg = GovernorConfig()  # all caps None → should resolve from tier
    resolved_xl = cfg.for_model("qwen/qwen3-coder-480b-a35b-instruct")
    assert resolved_xl.max_consecutive_tool_calls == TIER_DEFAULTS[
        ModelTier.XL
    ].max_consecutive_tool_calls
    assert resolved_xl.tool_cull_max == TIER_DEFAULTS[ModelTier.XL].tool_cull_max

    resolved_tiny = cfg.for_model("nvidia/nemotron-3-nano-30b-a3b")
    assert resolved_tiny.max_consecutive_tool_calls == TIER_DEFAULTS[
        ModelTier.TINY
    ].max_consecutive_tool_calls
    # XL must get a wider cap than TINY.
    assert (
        resolved_xl.max_consecutive_tool_calls
        > resolved_tiny.max_consecutive_tool_calls
    )


def test_governor_config_global_env_overrides_tier_defaults():
    """If an env-level value is set, every model uses it regardless of tier."""
    cfg = GovernorConfig(max_consecutive_tool_calls=12)  # global override
    resolved_tiny = cfg.for_model("nvidia/nemotron-3-nano-30b-a3b")
    resolved_xl = cfg.for_model("qwen/qwen3-coder-480b-a35b-instruct")
    assert resolved_tiny.max_consecutive_tool_calls == 12
    assert resolved_xl.max_consecutive_tool_calls == 12


def test_governor_config_per_model_override_beats_tier_and_env():
    cfg = GovernorConfig(
        max_consecutive_tool_calls=12,  # global env-level override
        overrides={
            "qwen3-coder-480b": ModelOverride(max_consecutive_tool_calls=99)
        },
    )
    resolved_xl = cfg.for_model("qwen/qwen3-coder-480b-a35b-instruct")
    resolved_other = cfg.for_model("qwen/qwen3-next-80b-a3b-instruct")
    assert resolved_xl.max_consecutive_tool_calls == 99
    assert resolved_other.max_consecutive_tool_calls == 12  # env wins


def test_governor_config_per_model_can_force_a_tier():
    """An override can pin the tier explicitly (e.g. force a tiny model into 'medium')."""
    cfg = GovernorConfig(
        overrides={
            "weird-model": ModelOverride(tier=ModelTier.LARGE),
        }
    )
    assert cfg.tier_for("weird-model") == ModelTier.LARGE
    resolved = cfg.for_model("weird-model")
    assert resolved.max_consecutive_tool_calls == TIER_DEFAULTS[
        ModelTier.LARGE
    ].max_consecutive_tool_calls


def test_load_governor_config_env_tier_caps_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "GOVERNOR_TIER_CAPS_JSON",
        '{"xl": {"max_consecutive_tool_calls": 99, "tool_cull_max": 70}}',
    )
    cfg = load_governor_config()
    xl_caps = cfg.tier_caps[ModelTier.XL]
    assert xl_caps.max_consecutive_tool_calls == 99
    assert xl_caps.tool_cull_max == 70
    # Other tiers untouched.
    assert (
        cfg.tier_caps[ModelTier.SMALL].max_consecutive_tool_calls
        == TIER_DEFAULTS[ModelTier.SMALL].max_consecutive_tool_calls
    )


def test_load_governor_config_unset_env_returns_none_caps(
    monkeypatch: pytest.MonkeyPatch,
):
    """Unset env caps remain None at process level so tier defaults can fill in."""
    for var in (
        "GOVERNOR_MAX_CONSECUTIVE_TOOL_CALLS",
        "GOVERNOR_MAX_TOTAL_TOOL_CALLS",
        "GOVERNOR_TOOL_CULL_MAX",
        "GOVERNOR_PLAN_MODE_FOR_WEAK",
        "GOVERNOR_TERMINATION_HINT_FOR_WEAK",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = load_governor_config()
    assert cfg.max_consecutive_tool_calls is None
    assert cfg.max_total_tool_calls is None
    assert cfg.tool_cull_max is None
    assert cfg.plan_mode_for_weak is None
    assert cfg.termination_hint_for_weak is None
