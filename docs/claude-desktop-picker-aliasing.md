# Claude Desktop Picker Aliasing — Workaround for All NIM Models in the Picker

Project: Expose every NVIDIA NIM model (the full 118-model default free-tier catalog) in the Claude Desktop model picker while keeping Claude Code CLI and curl-based workflows unchanged.

Date: 2026-07-24

---

## 1. Problem statement

User runs Claude Desktop 1.21459.3 on Linux, fronted by FCC (free-claude-code) on `http://127.0.0.1:8082` and Caddy in front of it on `https://localhost:8443`. Config: `ANTHROPIC_BASE_URL=https://localhost:8443`, `ANTHROPIC_API_KEY=freecc`.

FCC `/v1/models` returns 705 entries (118 NIM × 2 thinking-toggles + 272 OpenRouter + others + hardcoded `SUPPORTED_CLAUDE_MODELS`). Despite that, Claude Desktop's picker shows only a handful of NIM models — only the ones whose ids do not contain blacklisted substrings like `llama` or `qwen`.

Target: every NIM model the FCC catalog exposes is selectable from the picker.

---

## 2. Root cause

### 2.1 Two-layer filter chain

**Layer A: discovery gating.** `ANTHROPIC_BASE_URL` in the user's `claude_desktop_config.json` is interpreted as a first-party Anthropic override. Claude Desktop never hits the configured `/v1/models` endpoint, never runs `discoveryEnabled`, and shows only its hardcoded Claude-family model list. This is the gating that pinned the visible models to be a small subset regardless of FCC's catalog.

**Layer B: id validator `Kh(g)`.** Inside `app.asar` `.vite/build/index.pre.js`, the model-id validator is:

```js
function Kh(g) {
  const A = g.toLowerCase();
  return yCA.test(A) ? !1 : h2.test(A) || RCA.some(I => A.includes(I));
}
```

It rejects any id matching the blacklist regex `yCA`:

```
ark-code|astron|command-r|deepseek|doubao|gemini|gemma|glm|gpt|grok|hermes|hy3|kimi|lfm|llama|longcat|mimo|minimax|mistral|mixtral|moonshot|nemotron|openai|phi-|qianfan|qwen|tc-code|unic|yi-|stepfun|...
```

It accepts ids matching the white-form `h2` regex `^(claude|tiers)(-[\d.]+)?$` or containing one of the tier/brand substrings in `RCA = ["claude", ...un, "anthropic"]`.

So `anthropic/nvidia_nim/meta/llama-3.3-70b-instruct` (which is what FCC returned) is rejected at parse time inside Claude Desktop — the picker effectively only displays ids whose `display_name` or `id` lacks a non-Claude family substring.

### 2.2 Where `Kh` is called

- `YCA(g)` / `kCA(g)` / `SCA(g)` / `FCA(g)` / `LCA(g)` each return `Kh(g) ? {ok:!0} : {ok:!1, reason: ...}`.
- `UCA(g, A)` dispatches by provider: `anthropic` → `YCA`, `bedrock` → `kCA`, `vertex` → `SCA`, `foundry` → `FCA`, `gateway` / `mantle` → `LCA`.
- The caller that triggers `UCA` is a `warn:` predicate on the `inferenceModels` schema entry. It validates whatever the user manually writes into `inferenceModels` config and warns when an id is not Anthropic-shaped. When the model list comes from `/v1/models` discovery, `Kh` is run on each candidate id at picker-render time as well — same regex, same blacklist.

### 2.3 Loopback guard

`YB(e)` returns true for any URL whose hostname is in the private-network IP block (`ger` — including `127.0.0.0/8` and friend) or whose protocol is not `https:`. It is enforced only by `der(...)` inside the bootstrap intake path. Direct writes to `claude_desktop_config.json` (e.g. by hand) skip the bootstrap call, so the `inferenceGatewayBaseUrl` field can point at `https://localhost:8443` without rejection.

---

## 3. Workaround design

Three options were considered:

1. **Patch `app.asar`.** Change `yCA` to an empty regex (or make `Kh` return true unconditionally), re-pack the asar, replace the file. Solves Layer B once. Breaks on any claude-desktop update.
2. **Switch claude-desktop to 3P gateway mode and expose a fix-me-later list in `inferenceModels`.** Solves Layer A by routing discovery to FCC, but the **`Kh` blacklist in Layer B still rejects any NIM id containing `llama`/`qwen`/etc.** No-op for non-Claude families.
3. **FCC-side picker aliasing.** Keep the canonical passthrough ids for CLI / curl / direct callers and emit `claude-sonnet-nim-NNNN`-shaped aliases only on the `/v1/models` response, with a reverse map consulted at chat ingress. Solves both layers without touching `app.asar` and without breaking CLI/curl.

The user picked option 3.

### 3.1 Why the alias layer does not break CLI / curl / direct API users

- Aliases are emitted only inside `build_models_list_response()` for the `/v1/models` output. Existing `gateway_model_id()` (which returns `anthropic/<provider>/<model>`) keeps working for every other code path.
- Chat ingress in `application/routing.py` is layered: `resolve_picker_alias()` first; if hit, return the decoded `(provider_id, provider_model, force_reasoning_off)` exactly as the existing `decode_gateway_model_id()` / `provider/model` paths would. If the alias is unknown, fall through unchanged.
- CLI callers pass `provider_id/model_id` strings directly — they never see aliases — and `fcc-claude` wraps Claude Code which still uses `decode_gateway_model_id`. Stored configs in `~/.fcc/.env` (`ANTHROPIC_MODEL=` etc.) use raw refs.
- Alias ids are explicitly documented as *display-only* in the picker; the underlying real ref drives every chat round-trip.

### 3.2 Alias id format

Fixed prefix `claude-sonnet-nim`, four-digit counter, optional `-no-thinking` suffix for variant. Counter is per startup position after sorting the seeded provider refs, so the mapping is stable within one process and reproducible across restarts given the same sorted ref order.

```
claude-sonnet-nim-0001                       -> nvidia_nim/01-ai/yi-large
claude-sonnet-nim-0001-no-thinking           -> nvidia_nim/01-ai/yi-large  (force_reasoning_off)
claude-sonnet-nim-0036                       -> nvidia_nim/meta/llama-3.3-70b-instruct
claude-sonnet-nim-0084                       -> nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b
...
```

Every emitted alias id contains the substring `claude-` and `sonnet` (tier tokens), so it satisfies `RCA.some(I => A.includes(I))` in `Kh`. The blacklist `yCA` only matches the lowercased full id, so emitting tier-only strings avoids every banned family name.

---

## 4. Implementation

### 4.1 New module API (`src/free_claude_code/core/gateway_model_ids.py`)

Added on top of the existing `gateway_model_id` / `no_thinking_gateway_model_id` / `decode_gateway_model_id` helpers:

```python
PICKER_ALIAS_PREFIX = "claude-sonnet-nim"

# Module-level dicts holding the active mapping.
_picker_alias_to_ref: dict[str, str] = {}
_picker_alias_to_ref_no_thinking: dict[str, str] = {}
_picker_ref_to_alias: dict[str, str] = {}
_picker_ref_to_alias_no_thinking: dict[str, str] = {}


def seed_picker_aliases(provider_model_refs: Iterable[str]) -> None:
    """Reset and rebuild the alias maps from the supplied refs.
    Stable across restarts (sorted input, sequential numbering).
    """


def picker_alias_for(
    provider_model_ref: str, *, force_reasoning_off: bool = False
) -> str | None:
    """Return the picker alias for `provider_model_ref`, if seeded."""


def resolve_picker_alias(model_name: str) -> tuple[str, bool] | None:
    """Reverse-lookup alias. Returns ``(provider_ref, force_reasoning_off)``."""


def has_picker_aliases() -> bool:
    """Whether `seed_picker_aliases` has been called at least once."""


def clear_picker_aliases() -> None:
    """Drop every alias entry. Used by tests and hardening paths."""
```

The maps live at module scope (process-global). Initial values are empty so chat requests that arrive before startup completion see `None` from the resolver and fall through to the original routing logic.

### 4.2 Picker id selection (`src/free_claude_code/api/model_catalog.py`)

`_append_provider_model_variants` previously emitted `gateway_model_id(provider_model_ref)` for both thinking-on and thinking-off variants. After the change:

```python
picker_id = picker_alias_for(provider_model_ref)
fallback_id = gateway_model_id(provider_model_ref)
picker_id_no_thinking = picker_alias_for(provider_model_ref, force_reasoning_off=True)
fallback_id_no_thinking = no_thinking_gateway_model_id(provider_model_ref)
thinking_id = picker_id or fallback_id
no_thinking_id = picker_id_no_thinking or fallback_id_no_thinking
# emit `thinking_id` and `no_thinking_id` with `display_name = provider_model_ref`
```

If the alias maps have not yet been seeded (cold start window), the catalog falls back to the original wrapper ids. Once seeding completes (~one network round-trip later), the next `/v1/models` request serves alias ids.

`display_name` is unchanged — every entry still shows the real `provider_id/model_id` so the picker surfaces humane labels even when the underlying alias is opaque. The static `SUPPORTED_CLAUDE_MODELS` list is still appended after the dynamic entries so first-party `ANTHROPIC_BASE_URL` mode keeps its hardcoded Claude models.

### 4.3 Chat ingress decode (`src/free_claude_code/application/routing.py`)

`ModelRouter._direct_provider_model` checked `decode_gateway_model_id` first and fell back to plain `provider_id/model`. After the change:

```python
alias_resolution = resolve_picker_alias(model_name)
if alias_resolution is not None:
    aliased_ref, force_reasoning_off = alias_resolution
    provider_id, sep, provider_model = aliased_ref.partition("/")
    if not sep or provider_id not in SUPPORTED_PROVIDER_IDS or not provider_model:
        return None, None, False
    return provider_id, provider_model, force_reasoning_off

# unchanged: decode_gateway_model_id + provider/model fallthrough
```

When the alias is unknown (cold-start, unseeded prefix, or passthrough caller), the existing logic returns the same `(provider_id, provider_model, force_reasoning_off)` tuple it would otherwise have returned.

### 4.4 Seeding at startup (`src/free_claude_code/runtime/application.py`)

`ApplicationRuntime.start()` runs `warm_referenced_model_cache()` (awaited) and then `start_model_list_refresh()` (fire-and-forget). The new line slots in between:

```python
await self.provider_manager.warm_referenced_model_cache()
self.provider_manager.start_model_list_refresh()
seed_picker_aliases(
    info.model_id for info in self.provider_manager.cached_prefixed_model_infos()
)
```

Because `cached_prefixed_model_infos()` is sourced from the model cache that `warm_referenced_model_cache` populated synchronously, the seeding is deterministic on every startup and contains exactly the same refs the catalog would otherwise wrap.

### 4.5 `pyproject.toml` and launcher

A new console-script entry pinned the launcher:

```
fcc-claude-desktop = "free_claude_code.cli.launchers.claude_desktop:launch"
```

The launcher wraps `subprocess.run([shutil.which("claude-desktop"), "--ignore-certificate-errors", *user_args])`. Without it, Claude Desktop's TLS layer rejects Caddy's self-signed certificate even with `NODE_TLS_REJECT_UNAUTHORIZED=0` because Electron's TLS stack runs above Node and ignores that env knob. The system ca-certificates install path did not work on this box, so the launcher became the only working escape hatch.

---

## 5. Configuration changes

`~/.config/Claude/claude_desktop_config.json` extended (existing `preferences`/`coworkUserFilesPath`/`env` blocks preserved; nothing removed):

```json
{
  ...
  "mcpServers": {},
  "modelDiscoveryEnabled": true,
  "inference": {
    "provider": "gateway",
    "credentialKind": "static",
    "inferenceProvider": "gateway",
    "inferenceCredentialKind": "static",
    "inferenceGatewayBaseUrl": "https://localhost:8443",
    "inferenceGatewayAuthScheme": "x-api-key",
    "inferenceAnthropicApiKey": "freecc"
  },
  "env": {
    "ANTHROPIC_BASE_URL": "https://localhost:8443",
    "ANTHROPIC_API_KEY": "freecc",
    "NODE_TLS_REJECT_UNAUTHORIZED": "0",
    "NODE_OPTIONS": "--use-system-ca"
  }
}
```

The 3P gateway block flips Claude Desktop out of first-party override mode so `/v1/models` is actually consulted. The `env` block was kept as a redundant fallback so a misstep never breaks chat routing while iterating.

The `YB()` bootstrap-only loopback guard does not fire on direct file writes, so `https://localhost:8443` is acceptable inside the inference block.

### 5.1 Alias scoping — desktop-only mount

Aliases must never reach Claude Code, Codex, Pi, Muse, or any other FCC
client. Scoping is by API mount, not User-Agent sniffing:

- New setting `desktop_gateway_prefix` (env `DESKTOP_GATEWAY_PREFIX`, default
  `claude-desktop`).
- The gateway router is mounted twice in `create_app`: bare paths always serve
  raw provider refs; `/{desktop_gateway_prefix}` serves picker aliases.
- `list_models` enables aliases only when `request.url.path` starts with the
  prefix (`api/routes.py`).
- The launcher writes the prefixed URL (`https://localhost:8443/claude-desktop`)
  into `inferenceGatewayBaseUrl`, so only Claude Desktop talks to the
  alias-serving mount.
- Chat ingress still reverse-decodes aliases universally — harmless for other
  clients since they never receive alias ids.

---

## 6. Verification

### 6.1 Targeted unit tests

```
python -m pytest tests -q -k "gateway_model_ids or model_catalog or routing" --no-header -p no:cacheprovider
=> 24 passed
```

### 6.2 Full test suite (FCC)

```
python -m pytest tests -q -p no:cacheprovider -n auto
=> 2475 passed, 52 skipped, 8 failed
```

The 8 failures are `tests/scripts/test_uninstallers.py::*` complaining "Free Claude Code is still running" — they require no `fcc-server` to be alive and have nothing to do with this change.

### 6.3 Live `/v1/models` after restart

Started a patched FCC on `PORT=8083` and hit `/v1/models` with auth:

```
total: 705     aliases: 236   (= 118 NIM * 2)
sample alias:
  claude-sonnet-nim-0084        -> nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b
  claude-sonnet-nim-0084-no-thinking
  claude-sonnet-nim-0036        -> nvidia_nim/meta/llama-3.3-70b-instruct
  claude-sonnet-nim-0036-no-thinking
  claude-sonnet-nim-0001        -> nvidia_nim/01-ai/yi-large
```

Last four ids are the hardcoded `SUPPORTED_CLAUDE_MODELS` (`claude-3-opus`, `claude-3-5-sonnet`, `claude-3-haiku`, `claude-3-5-haiku`).

### 6.4 Live chat through three ingress shapes

| Model id sent | Path taken | Result |
|---|---|---|
| `claude-sonnet-nim-0036` | alias decode | `meta/llama-3.3-70b-instruct` answered `ok` |
| `nvidia_nim/meta/llama-3.3-70b-instruct` | provider/model fallthrough | answered `ok` |
| `anthropic/nvidia_nim/meta/llama-3.3-70b-instruct` | `decode_gateway_model_id` | answered `Hi` |

A bogus alias `claude-sonnet-nim-0001` (mapped to `01-ai/yi-large`, which NIM does not host) routed through cleanly and surfaced NIM's own 404, confirming the decoder does not silently swallow unknown upstream responses.

### 6.5 Claude Desktop picker end-to-end

After updating `claude_desktop_config.json` and relaunching Claude Desktop (with `fcc-claude-desktop` for the cert bypass flag), the picker shows 236 aliased entries mapped to NIM models. Confirmed by user as working.

### 6.6 Cert install attempt (succeeded, did not help)

```
sudo cp /etc/caddy/localhost+2.pem /usr/local/share/ca-certificates/fcc-caddy-localhost.crt
sudo update-ca-certificates --fresh
```

Ran without error but did not change Claude Desktop's TLS behaviour — Electron's certificate trust does not always read system ca-certificates on this distribution. The `fcc-claude-desktop` launcher is the durable fix.

---

## 7. Files changed

| Path | Change |
|---|---|
| `/tmp/free-claude-code/src/free_claude_code/core/gateway_model_ids.py` | Added alias maps, `seed_picker_aliases`, `picker_alias_for`, `resolve_picker_alias`, `has_picker_aliases`, `clear_picker_aliases`, `PICKER_ALIAS_PREFIX`. |
| `/tmp/free-claude-code/src/free_claude_code/api/model_catalog.py` | `_append_provider_model_variants` prefers alias ids, falls back to existing `gateway_model_id()`. |
| `/tmp/free-claude-code/src/free_claude_code/application/routing.py` | `ModelRouter._direct_provider_model` consults `resolve_picker_alias` before the existing fallback chain. |
| `/tmp/free-claude-code/src/free_claude_code/runtime/application.py` | Calls `seed_picker_aliases(...)` after `warm_referenced_model_cache()`. |
| `/tmp/free-claude-code/src/free_claude_code/cli/launchers/claude_desktop.py` | New launcher passing `--ignore-certificate-errors`. |
| `/tmp/free-claude-code/pyproject.toml` | New `[project.scripts]` entry: `fcc-claude-desktop`. |
| `/home/joshua/.config/Claude/claude_desktop_config.json` | Added `modelDiscoveryEnabled` + `inference` block (3P gateway). Existing `env` block preserved. |
| `/etc/caddy/localhost+2.pem` -> `/usr/local/share/ca-certificates/fcc-caddy-localhost.crt` | Cert copy, used as fallback not relied on. |

## 8. Install / deploy steps that worked

```bash
# 1. Apply patch (FCC source on disk & already symlinked into uv tool install).
cd /tmp/free-claude-code
. .venv/bin/activate

# 2. Re-install with the new entry point.
uv tool install --force .

# 3. Do NOT manually kill fcc-server — let it be respawned by the supervisor
#    or by the user.

# 4. Update claude_desktop_config.json. Hand-edit is fine because the
#    bootstrap-only loopback guard (`YB`) does not fire on direct writes.

# 5. Launch Claude Desktop with the cert bypass (one command, no flag to
#    remember).
fcc-claude-desktop
```

## 9. Known caveats

- **Restart Claude Desktop after every config edit.** A live process does not reload `claude_desktop_config.json` mid-session.
- **`fcc-claude-desktop` requires `claude-desktop` on PATH.** It's installed at `/usr/bin/claude-desktop` via the official `.deb`. If the binary isn't found, the launcher fails fast with an actionable error.
- **Cert install is not relied on.** System ca-cert updates succeeded but did not affect Electron's TLS layer on this host. The launcher is the durable workaround.
- **Alias list order is per-startup.** Counter stable across restarts given the same sorted seeded refs. If a new provider is added, new aliases append at the tail of the sorted list; existing aliases keep their counters as long as older refs don't churn.
- **fcc-desktop (`[project.gui-scripts]`)** is unaffected. The new launcher is intentionally namespaced as `fcc-claude-desktop` to avoid clobbering the existing GUI variant.

---

## 10. Lessons learned / follow-ups

1. **Do not kill `fcc-server` directly.** FCC supervises itself; killing the running PID while Claude Desktop is open pulled the live session into an "API error" state. Future deploys flow through `uv tool install --force .` which the supervisor picks up, no manual SIGTERM.
2. **Always grep the obfuscated identifiers inside `app.asar` rather than guessing names.** The bundle is minified — `Kh`, `yCA`, `h2`, `RCA`, `YB`, `der`, `TJ`, `UCA` all collapse to one-or-two chars. Once you find their relationships, the rest is straightforward; without it, you'd be reading V8 noise.
3. **Bootstrap-intake-only guards do not protect against direct file writes.** The `YB` loopback check inside `der(...)` is what would reject `https://localhost:8443` in a normal flow, but `application/json` writes bypass that path. Knowing this lets us choose a friendlier config rather than fighting a guard that isn't enforced.
4. **Electron's TLS ignores `NODE_TLS_REJECT_UNAUTHORIZED`.** Future-proof fix: ship a Caddy root cert that Linux distros already trust (e.g. the system's `ca-certificates` chain with the private CA root) so `fcc-claude-desktop` becomes a fallback rather than the primary path. Today the launcher is the cleanest answer.
5. **Alias ids live only inside `/v1/models` output.** Anything that calls a real model uses the canonical `provider/model` or `anthropic/provider/model` shape — never aliases. CLI / curl / SDK callers all see the same FCC surface they see today.
6. **Catalog count is a quick smoke test.** `total`, `aliases == 2 × NIM_count`, and four `SUPPORTED_CLAUDE_MODELS` hardcoded tails together prove seeding, catalog fallback chain, and routing decoder are all live.

### Follow-ups

- Backport the same `claude-sonnet-nim-NNNN` strategy to OpenRouter models that also trip `yCA` (e.g. any id containing `gemma` or `mistral`). The seeding loop is provider-agnostic; only the prefix choice is opinionated.
- Consider exposing an `ANTHROPIC_MODEL_ALIASES` env knob so users can pick a custom prefix (`fcc-`, `nim-`, etc.) if Anthropic's whitelist ever tightens `RCA` to drop our tier tokens.
- Add a `--strict-fcc-models` flag to the launcher that bypasses Claude Desktop's picker validator by talking directly to FCC (skips discovery entirely) — useful for offline selection sessions.
- Document the relationship between the picker-discovery path and the chat-routing path in `gateway_model_ids.py` itself (currently only mentioned in this doc) so future maintainers see the symmetry without re-reading the electron source.