# Plan: Per-Tier Model Mapping (Opus/Sonnet/Haiku)

## Problem
Currently, **all** Claude model requests (regardless of whether Claude Code sends `claude-opus-4-...`, `claude-sonnet-4-...`, or `claude-haiku-3-...`) are mapped to a **single** backend model via the `MODEL` env var. This means the same model handles both the main agent (opus-tier) and lightweight subagent tasks (haiku-tier), which is suboptimal.

## Goal
Allow users to configure **separate** provider+model combos for each Claude tier:
- `MODEL_OPUS` → e.g. `open_router/deepseek/deepseek-r1` (heavy reasoning)
- `MODEL_SONNET` → e.g. `nvidia_nim/meta/llama-3.3-70b-instruct` (balanced)
- `MODEL_HAIKU` → e.g. `lmstudio/qwen2.5-7b` (fast/cheap subagent tasks)
- `MODEL` → fallback when a tier-specific var is not set

Each tier can use a **different provider**, so the system must resolve the correct provider per-request rather than using a single global provider.

## Architecture Changes

### 1. `config/settings.py` — Add per-tier model fields

- Add three new optional fields: `model_opus`, `model_sonnet`, `model_haiku` (all `str | None`, default `None`)
- Env vars: `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU`
- Apply the same `validate_model_format` validation to them (when set)
- Add a helper method `resolve_model(claude_model_name: str) -> str` that:
  1. Classifies the incoming Claude model name into a tier (opus/sonnet/haiku) based on substring matching
  2. Returns the tier-specific model if set, otherwise the fallback `MODEL`
- Add helper properties: `resolve_provider_type(model_string: str) -> str` and `resolve_model_name(model_string: str) -> str` — same logic as existing `provider_type`/`model_name` but for arbitrary model strings

### 2. `api/models/anthropic.py` — Tier-aware model mapping

- Update `MessagesRequest.map_model()` to call `settings.resolve_model()` instead of always using `settings.model_name`
- Store the full resolved model string (with provider prefix) in a new field so the route handler can determine which provider to use
- Update `TokenCountRequest.validate_model_field()` similarly

### 3. `api/dependencies.py` — Multi-provider support

- Instead of a single `_provider`, maintain a **provider registry** (`dict[str, BaseProvider]`) keyed by provider type
- `_create_provider(provider_type, settings)` → creates a provider for the given type
- `get_provider_for_request(request)` → resolves the correct provider based on the request's resolved model
- Keep backward compatibility: `get_provider()` returns the default (fallback MODEL) provider for health checks, root endpoint, etc.

### 4. `api/routes.py` — Use per-request provider resolution

- Update `create_message()` to resolve the provider from the request's resolved model string rather than using the global singleton

### 5. `.env.example` — Document new env vars

- Add `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU` with comments explaining the fallback behavior

### 6. Tests

- **`tests/config/test_config.py`**: Test `resolve_model()` with various Claude model names, tier overrides, and fallback behavior
- **`tests/api/test_models_validators.py`**: Test tier-aware model mapping in `MessagesRequest`
- **`tests/api/test_dependencies.py`**: Test multi-provider registry creation and resolution

## Tier Classification Logic

Claude Code sends model names like:
- Opus: `claude-opus-4-...`, `claude-3-opus`, `claude-3-opus-20240229`
- Sonnet: `claude-sonnet-4-...`, `claude-3-sonnet`, `claude-3-5-sonnet-20241022`
- Haiku: `claude-haiku-...`, `claude-3-haiku`, `claude-3-haiku-20240307`, `claude-3-5-haiku-20241022`

Classification: check if the model name contains `opus`, `sonnet`, or `haiku` (case-insensitive). Default to sonnet tier if unclear (most common tier).

## File Change Summary

| File | Change |
|---|---|
| `config/settings.py` | Add `model_opus`, `model_sonnet`, `model_haiku` fields + `resolve_model()` + validation |
| `api/models/anthropic.py` | Use `resolve_model()` in `map_model()`, store `resolved_model_full` (with provider prefix) |
| `api/dependencies.py` | Provider registry, per-request provider resolution |
| `api/routes.py` | Resolve provider per-request |
| `.env.example` | Document `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU` |
| `tests/config/test_config.py` | Tests for `resolve_model()` |
| `tests/api/test_models_validators.py` | Tests for tier-aware mapping |
| `tests/api/test_dependencies.py` | Tests for multi-provider resolution |

## Key Design Decisions

1. **Fallback chain**: `MODEL_OPUS` → `MODEL` (not `MODEL_OPUS` → `MODEL_SONNET` → `MODEL`). Each tier falls back directly to `MODEL`.
2. **Provider per-request, not per-tier**: Since each tier's model string includes the provider prefix, the provider is resolved from the full model string at request time. No separate "provider per tier" config needed.
3. **Lazy provider creation**: Providers are created on first use and cached in the registry. If a user only configures `MODEL_OPUS` with OpenRouter and `MODEL` with NIM, only those two providers get instantiated.
4. **Backward compatible**: If no `MODEL_*` vars are set, behavior is identical to today — all requests map to `MODEL`.
5. **Cleanup**: All providers in the registry are cleaned up on shutdown.

## Residual Risks
- None anticipated. The fallback to `MODEL` ensures existing setups work unchanged.
