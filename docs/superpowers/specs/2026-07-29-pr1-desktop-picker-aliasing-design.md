# PR1: Claude Desktop Picker Aliasing + Launcher

Date: 2026-07-29
Project: free-claude-code v4.13.1 → 4.14.0

## Goal

Route every NIM entry through Claude Desktop's model picker without modifying
the obfuscated validator inside `app.asar`. Emit `claude-sonnet-nim-NNNN`-shaped
picker ids on `/v1/models`; resolve them to canonical `provider/model` ref on
chat ingress. Ship a `fcc-claude-desktop` launcher and `fcc-claude-desktop
--configure` helper so install auto-writes user's `claude_desktop_config.json`.

## Background

Source documents:

- `docs/claude-desktop-picker-aliasing.md` (full design, earlier lost work)
- `docs/fcc-integration-analysis.md` §3 (model routing + aliases row)
- `docs/integration-issues-analysis.md` Issue 6 (capability advertisement)

## Architecture

Five thin layers; failure-isolated:

1. **Mapping layer** (`core/gateway_model_ids.py`)
   Pure functions. Module-scope dicts. Deterministic sorted-input → counter.
   Cold-start state (empty maps) → resolver returns `None` → fallback wins.

2. **Catalog layer** (`api/model_catalog.py`)
   `_append_provider_model_variants` prefers aliased id; falls back to existing
   `gateway_model_id()`. `display_name` stays as `provider/model_ref`.

3. **Routing layer** (`application/routing.py`)
   `ModelRouter._direct_provider_model` checks `resolve_picker_alias` first;
   falls through to `decode_gateway_model_id` / provider/model chain.

4. **Lifecycle layer** (`runtime/application.py`)
   After `warm_referenced_model_cache()` succeeds, call `seed_picker_aliases`
   exactly once with the sorted refs `cached_prefixed_model_infos()` exposes.

5. **OS layer** (`cli/launchers/claude_desktop.py`)
   Two modes:
   - Default: launch `claude-desktop --ignore-certificate-errors`.
   - `--configure`: read/merge `~/.config/Claude/claude_desktop_config.json`
     via stdlib `json` (no jq dependency). Idempotent.
   - `--unconfigure`: reverse the merge.

## Behavior — alias resolution

Stable counter given sorted ref list:

```
nvidia_nim/01-ai/yi-large                   -> claude-sonnet-nim-0001
nvidia_nim/01-ai/yi-large (no-thinking)     -> claude-sonnet-nim-0001-no-thinking
nvidia_nim/meta/llama-3.3-70b-instruct      -> claude-sonnet-nim-0002
...
```

Aliases always contain `claude-` and `sonnet` tokens; satisfy
`RCA.some(I => A.includes(I))` in the desktop validator. Blacklist substrings
(`llama`, `qwen`, ...) absent in ids.

## JSON merge contract

`--configure` writes the following keys idempotently under existing file:

```json
{
  "modelDiscoveryEnabled": true,
  "inference": {
    "provider": "gateway",
    "credentialKind": "static",
    "inferenceProvider": "gateway",
    "inferenceCredentialKind": "static",
    "inferenceGatewayBaseUrl": "https://localhost:8443",
    "inferenceGatewayAuthScheme": "x-api-key",
    "inferenceAnthropicApiKey": "freecc"
  }
}
```

Preserves `preferences`, `coworkUserFilesPath`, `mcpServers`, `env`. Does not
clobber unknown keys. `env` may carry `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY`
as fallback.

`--unconfigure` removes only the keys above; preserves every other key byte-for-byte.

## Install script flow

`scripts/install.sh` / `install.ps1`: after `uv tool install --force .`
succeeds and before surfacing completion, invoke the Python helper. Cross-
platform: `python3 -m free_claude_code.cli.launchers.claude_desktop --configure`
on POSIX, `\1 -m free_claude_code.cli.launchers.claude_desktop --configure` on
Windows. Helper is in the just-installed tool so it's reachable. Fail loudly if
config path unreachable; silent if file absent.

Best-effort cert copy into `ca-certificates` after configure — Electron
ignores it on some distros, so log only.

## Tests (TDD)

- `tests/core/test_gateway_model_ids.py`: seed/lookup round-trip, no-thinking
  variant, unknown alias returns None, clear resets state, counter deterministic
  from sorted input.
- `tests/api/test_model_catalog.py`: alias id selected when seeded, fallback
  to wrapper when unseeded, `display_name` equals real ref.
- `tests/application/test_routing.py`: alias decode hits before gateway id,
  before `provider/model` passthrough, bogus alias surfaces upstream 404.
- `tests/cli/test_claude_desktop_launcher.py`: subprocess args, binary-not-found,
  exit-code passthrough, JSON merge idempotency, unconfigure round-trip.
- `tests/scripts/test_install_invokes_configure.py`: install script argparse
  contains `python3 -m free_claude_code.cli.launchers.claude_desktop --configure`.

## Version

`pyproject.toml` version `4.13.1` → `4.14.0` (MINOR — backward-compatible
feature add). Same commit as production diff; `uv lock` updated.

## Out of scope (PR2+)

- `/v1/telemetry` 200 / payload
- Capability flag surfacing
- MCP provider implementation
- Files API
- Voice endpoint
- Document upload

## Residual risks

- Counter stability across restarts depends on `cached_prefixed_model_infos()`
  returning same sorted refs post-restart. If FCC adds a ref-sorted seam later
  that changes order, alias numbering shifts. Mitigated: alias map is
  deterministic, never reused — but callers should treat alias ids as
  ephemeral labels.
- Electron TLS: `--ignore-certificate-errors` is per-binary; updating
  `claude-desktop` binary drops it. Re-install required. Documented in
  launcher help text.
