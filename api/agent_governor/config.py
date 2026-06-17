"""Environment-driven configuration for the agent governor.

Cap fields are ``int | None``: ``None`` means "delegate to the tier defaults
for the model in question". ``load_governor_config()`` populates them from
env (when set) or leaves them ``None`` so that :meth:`GovernorConfig.for_model`
can fall back to size-tier defaults.

Resolution order (highest precedence → lowest):
  1. Per-model override (``GOVERNOR_OVERRIDES_JSON``)
  2. Process-level env value (e.g. ``GOVERNOR_MAX_CONSECUTIVE_TOOL_CALLS``)
  3. Tier defaults (from :mod:`api.agent_governor.tiers`,
     overridable via ``GOVERNOR_TIER_CAPS_JSON``)
  4. Hardcoded fallback if everything else is missing
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .tiers import (
    TIER_DEFAULTS,
    ModelTier,
    TierCaps,
    caps_for_tier,
    detect_tier,
)


_DEFAULT_WEAK_PATTERNS = (
    "qwen",
    "nemotron",
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-v",
    "kimi",
    "moonshot",
    "gpt-oss",
    "glm",
    "mistral",
    "llama",
    "wafer",
    "ring",
    "minimax",
    "gemma",
    "hermes",
    "command",
    "yi-",
)
_DEFAULT_STRONG_PATTERNS = (
    "claude-",
    "gpt-5",
    "gpt-4",
    "o1-",
    "o3-",
    "o4-",
    "grok-",
    "gemini-",
)


def _bool_env(name: str, default: bool | None) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _int_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid {} value: {!r} — using default {}", name, raw, default
        )
        return default


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _json_env(name: str) -> dict[str, Any]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Invalid {} JSON: {} — ignoring overrides", name, exc.msg
        )
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            "{} must decode to a JSON object — ignoring overrides", name
        )
        return {}
    return parsed


@dataclass(slots=True)
class ModelOverride:
    """Per-model configuration override (subset of GovernorConfig)."""

    enabled: bool | None = None
    tool_cull_enabled: bool | None = None
    tool_cull_max: int | None = None
    max_consecutive_tool_calls: int | None = None
    max_total_tool_calls: int | None = None
    loop_repeat_threshold: int | None = None
    plan_mode: bool | None = None
    termination_hint: bool | None = None
    tier: ModelTier | None = None  # force a specific tier for this model


@dataclass(slots=True)
class GovernorConfig:
    """Resolved governor configuration for one request.

    When :meth:`for_model` runs, this returns a new instance with all
    Optional cap fields filled in from tier defaults / env / overrides.
    The "process-level" instance (loaded from env) may carry ``None``
    values; the "per-model" instance never does.
    """

    enabled: bool = True
    tool_cull_enabled: bool = True
    tool_cull_max: int | None = None
    max_consecutive_tool_calls: int | None = None
    max_total_tool_calls: int | None = None
    loop_repeat_threshold: int = 3
    plan_mode_for_weak: bool | None = None
    termination_hint_for_weak: bool | None = None
    weak_model_patterns: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_WEAK_PATTERNS
    )
    strong_model_patterns: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_STRONG_PATTERNS
    )
    overrides: dict[str, ModelOverride] = field(default_factory=dict)
    tier_caps: dict[ModelTier, TierCaps] = field(
        default_factory=lambda: dict(TIER_DEFAULTS)
    )

    def is_strong_model(self, model_id: str) -> bool:
        lowered = model_id.lower()
        return any(pat in lowered for pat in self.strong_model_patterns)

    def is_weak_model(self, model_id: str) -> bool:
        lowered = model_id.lower()
        return any(pat in lowered for pat in self.weak_model_patterns)

    def tier_for(self, model_id: str) -> ModelTier:
        """Return the size tier this config will use for ``model_id``."""
        if self.is_strong_model(model_id):
            return ModelTier.FRONTIER
        # Per-model override may pin a tier explicitly.
        override = self._lookup_override(model_id)
        if override and override.tier is not None:
            return override.tier
        return detect_tier(model_id)

    def for_model(self, model_id: str) -> GovernorConfig:
        """Return effective config for one model with all caps resolved."""
        override = self._lookup_override(model_id)
        tier = self.tier_for(model_id)
        tier_caps = self.tier_caps.get(tier) or caps_for_tier(tier)

        def resolve_int(*candidates: int | None) -> int:
            for value in candidates:
                if value is not None:
                    return value
            return 0

        def resolve_bool(*candidates: bool | None) -> bool:
            for value in candidates:
                if value is not None:
                    return value
            return False

        ovr = override
        return GovernorConfig(
            enabled=resolve_bool(
                ovr.enabled if ovr else None, self.enabled, True
            ),
            tool_cull_enabled=resolve_bool(
                ovr.tool_cull_enabled if ovr else None,
                self.tool_cull_enabled,
                True,
            ),
            tool_cull_max=resolve_int(
                ovr.tool_cull_max if ovr else None,
                self.tool_cull_max,
                tier_caps.tool_cull_max,
            ),
            max_consecutive_tool_calls=resolve_int(
                ovr.max_consecutive_tool_calls if ovr else None,
                self.max_consecutive_tool_calls,
                tier_caps.max_consecutive_tool_calls,
            ),
            max_total_tool_calls=resolve_int(
                ovr.max_total_tool_calls if ovr else None,
                self.max_total_tool_calls,
                tier_caps.max_total_tool_calls,
            ),
            loop_repeat_threshold=ovr.loop_repeat_threshold
            if ovr and ovr.loop_repeat_threshold is not None
            else self.loop_repeat_threshold,
            plan_mode_for_weak=resolve_bool(
                ovr.plan_mode if ovr else None,
                self.plan_mode_for_weak,
                tier_caps.plan_mode,
            ),
            termination_hint_for_weak=resolve_bool(
                ovr.termination_hint if ovr else None,
                self.termination_hint_for_weak,
                tier_caps.termination_hint,
            ),
            weak_model_patterns=self.weak_model_patterns,
            strong_model_patterns=self.strong_model_patterns,
            overrides=self.overrides,
            tier_caps=self.tier_caps,
        )

    def _lookup_override(self, model_id: str) -> ModelOverride | None:
        if not self.overrides:
            return None
        if model_id in self.overrides:
            return self.overrides[model_id]
        best_key = self._best_override_key(model_id)
        return self.overrides.get(best_key) if best_key else None

    def _best_override_key(self, model_id: str) -> str | None:
        """Find the longest configured key that is a substring of model_id."""
        candidates = [k for k in self.overrides if k and k in model_id]
        if not candidates:
            return None
        return max(candidates, key=len)


def _parse_tier_caps_overrides(raw: dict[str, Any]) -> dict[ModelTier, TierCaps]:
    """Apply env-supplied per-tier overrides on top of TIER_DEFAULTS."""
    merged: dict[ModelTier, TierCaps] = dict(TIER_DEFAULTS)
    for tier_name, payload in raw.items():
        if not isinstance(tier_name, str) or not isinstance(payload, dict):
            continue
        try:
            tier = ModelTier(tier_name.lower())
        except ValueError:
            logger.warning(
                "Unknown tier {!r} in GOVERNOR_TIER_CAPS_JSON — skipping",
                tier_name,
            )
            continue
        base = merged.get(tier, TIER_DEFAULTS[ModelTier.UNKNOWN])
        merged[tier] = TierCaps(
            max_consecutive_tool_calls=int(
                payload.get("max_consecutive_tool_calls", base.max_consecutive_tool_calls)
            ),
            max_total_tool_calls=int(
                payload.get("max_total_tool_calls", base.max_total_tool_calls)
            ),
            tool_cull_max=int(
                payload.get("tool_cull_max", base.tool_cull_max)
            ),
            plan_mode=bool(payload.get("plan_mode", base.plan_mode)),
            termination_hint=bool(
                payload.get("termination_hint", base.termination_hint)
            ),
        )
    return merged


def load_governor_config() -> GovernorConfig:
    """Build a GovernorConfig from environment variables (process-level)."""
    cfg = GovernorConfig(
        enabled=_bool_env("GOVERNOR_ENABLED", True) or False,
        tool_cull_enabled=_bool_env("GOVERNOR_TOOL_CULL_ENABLED", True) or False,
        tool_cull_max=_int_env("GOVERNOR_TOOL_CULL_MAX", None),
        max_consecutive_tool_calls=_int_env(
            "GOVERNOR_MAX_CONSECUTIVE_TOOL_CALLS", None
        ),
        max_total_tool_calls=_int_env("GOVERNOR_MAX_TOTAL_TOOL_CALLS", None),
        loop_repeat_threshold=_int_env("GOVERNOR_LOOP_REPEAT_THRESHOLD", 3) or 3,
        plan_mode_for_weak=_bool_env("GOVERNOR_PLAN_MODE_FOR_WEAK", None),
        termination_hint_for_weak=_bool_env(
            "GOVERNOR_TERMINATION_HINT_FOR_WEAK", None
        ),
        weak_model_patterns=_csv_env(
            "GOVERNOR_WEAK_MODEL_PATTERNS", _DEFAULT_WEAK_PATTERNS
        ),
        strong_model_patterns=_csv_env(
            "GOVERNOR_STRONG_MODEL_PATTERNS", _DEFAULT_STRONG_PATTERNS
        ),
    )
    cfg.tier_caps = _parse_tier_caps_overrides(
        _json_env("GOVERNOR_TIER_CAPS_JSON")
    )
    overrides_raw = _json_env("GOVERNOR_OVERRIDES_JSON")
    overrides: dict[str, ModelOverride] = {}
    for model_id, payload in overrides_raw.items():
        if not isinstance(model_id, str) or not isinstance(payload, dict):
            continue
        tier_value = payload.get("tier")
        tier_enum: ModelTier | None
        if isinstance(tier_value, str):
            try:
                tier_enum = ModelTier(tier_value.lower())
            except ValueError:
                logger.warning(
                    "Unknown tier {!r} in GOVERNOR_OVERRIDES_JSON for {!r}",
                    tier_value,
                    model_id,
                )
                tier_enum = None
        else:
            tier_enum = None
        overrides[model_id] = ModelOverride(
            enabled=payload.get("enabled"),
            tool_cull_enabled=payload.get("tool_cull_enabled"),
            tool_cull_max=payload.get("tool_cull_max"),
            max_consecutive_tool_calls=payload.get("max_consecutive_tool_calls"),
            max_total_tool_calls=payload.get("max_total_tool_calls"),
            loop_repeat_threshold=payload.get("loop_repeat_threshold"),
            plan_mode=payload.get("plan_mode"),
            termination_hint=payload.get("termination_hint"),
            tier=tier_enum,
        )
    cfg.overrides = overrides
    return cfg
