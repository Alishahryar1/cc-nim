# Free Claude Code — Project Blueprint
**Version:** 2.0.0 · **Updated:** 2026-06-15 · **Python:** 3.14.0
**Live:** https://free-claude-code-main-ebon.vercel.app · **Repo:** https://github.com/dnzengou/free-claude-code

---

## Executive Summary

Drop-in Anthropic Messages API proxy that routes Claude Code CLI traffic to 18 provider backends. Keeps Claude Code's client-side protocol stable while swapping the underlying model. Ships with a local Admin UI, Discord/Telegram bot wrapper, and optional voice-note transcription.

---

## Architecture

```
Claude Code CLI / VS Code / JetBrains ACP
        │  Anthropic Messages API
        ▼
┌─────────────────────────────────────────┐
│           FastAPI Proxy (server.py)     │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │  /admin  │  │  /v1/messages        │ │
│  │  Admin   │  │  /v1/models          │ │
│  │  UI      │  │  /v1/messages/count  │ │
│  └──────────┘  └──────────┬───────────┘ │
│                           │             │
│  ┌────────────────────────▼───────────┐ │
│  │         Model Router               │ │
│  │  Opus → MODEL_OPUS                 │ │
│  │  Sonnet → MODEL_SONNET             │ │
│  │  Haiku → MODEL_HAIKU               │ │
│  │  * → MODEL (fallback)              │ │
│  └────────────────────────┬───────────┘ │
└───────────────────────────┼─────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 AnthropicMessages    OpenAICompat         Local
 Transport            Transport            Providers
 (DeepSeek, Kimi,    (NIM, OpenRouter,    (LM Studio,
  Wafer, Fireworks,   Gemini, Groq,        llama.cpp,
  Z.ai)              Cerebras, Mistral,    Ollama)
                      OpenCode, Codestral)
```

---

## Module Map

| Package | Responsibility |
|---|---|
| `server.py` | ASGI entry point — `create_asgi_app()` |
| `api/` | FastAPI routes, admin UI, services, model routing, request optimizations |
| `core/anthropic/` | Shared Anthropic protocol: SSE, content, conversion, thinking, tokens, tools |
| `core/` | Rate limiting, trace/logging utilities |
| `providers/` | Provider transports (18 backends), registry, base classes, error mapping |
| `config/` | Settings (`pydantic-settings`), provider catalog, logging config, paths |
| `messaging/` | Discord/Telegram platforms, session trees, voice transcription |
| `cli/` | `fcc-server`, `fcc-claude`, `fcc-init` entry points; process registry |
| `tests/` | 1443 unit, contract, and integration tests |

---

## Provider Registry

| ID | Transport | Auth |
|---|---|---|
| `nvidia_nim` | OpenAI compat | `NVIDIA_NIM_API_KEY` |
| `open_router` | OpenAI compat | `OPENROUTER_API_KEY` |
| `gemini` | OpenAI compat | `GEMINI_API_KEY` |
| `mistral` | OpenAI compat | `MISTRAL_API_KEY` |
| `mistral_codestral` | OpenAI compat | `CODESTRAL_API_KEY` |
| `opencode` | OpenAI compat | `OPENCODE_API_KEY` |
| `opencode_go` | OpenAI compat | `OPENCODE_API_KEY` |
| `groq` | OpenAI compat | `GROQ_API_KEY` |
| `cerebras` | OpenAI compat | `CEREBRAS_API_KEY` |
| `openai` | OpenAI compat | `OPENAI_API_KEY` |
| `deepseek` | Anthropic native | `DEEPSEEK_API_KEY` |
| `kimi` | Anthropic native | `KIMI_API_KEY` |
| `wafer` | Anthropic native | `WAFER_API_KEY` |
| `fireworks` | Anthropic native | `FIREWORKS_API_KEY` |
| `zai` | Anthropic native | `ZAI_API_KEY` |
| `lmstudio` | OpenAI compat (local) | `LM_STUDIO_BASE_URL` |
| `llamacpp` | Anthropic native (local) | `LLAMACPP_BASE_URL` |
| `ollama` | OpenAI compat (local) | `OLLAMA_BASE_URL` |

---

## Admin UI

Local-only (loopback guard), served at `/admin`.

| File | Purpose |
|---|---|
| `api/admin_static/index.html` | Shell: sidebar, views, toast region, action bar |
| `api/admin_static/fcc.css` | Design system, animations, toast, skeleton, responsive, metrics |
| `api/admin_static/admin.js` | Toast, auto-refresh, keyboard shortcuts, copy, dirty guard |

**Features shipped:**
- Toast notification system (ok / warn / error / info, auto-dismiss 4.2 s)
- Provider status auto-refresh every 30 s with pulsing indicator
- `Ctrl+S` → validate · `Ctrl+Enter` → apply
- Copy-to-clipboard on non-secret text fields
- Skeleton loading cards on boot
- `beforeunload` unsaved-changes browser guard
- 6 CSS keyframe animations; accent glow on focused inputs
- Full mobile-responsive layout
- Dark/light theme toggle — `localStorage` persistence, sun/moon icons, full CSS variable override
- Metrics tab — summary cards, latency sparkline, per-request table with inline bars; `GET /admin/api/metrics`
- One-click API key validation — "Validate key" button per non-local provider card; `POST /admin/api/providers/{id}/validate`; 3-state result: key valid / auth OK (completion failed) / key invalid

---

## Deployment

| Target | Method | URL |
|---|---|---|
| **Vercel (production)** | `vercel --prod` / push to `main` | https://free-claude-code-main-ebon.vercel.app |
| **Local** | `uv run uvicorn server:app --host 0.0.0.0 --port 8082` | http://localhost:8082 |
| **Docker** | `docker build -t fcc . && docker run -p 8082:8082 fcc` | http://localhost:8082 |

**Required env vars on Vercel dashboard:**
```
MODEL=<provider>/<model>          # e.g. nvidia_nim/nvidia/nemotron-3-super-120b-a12b
ANTHROPIC_AUTH_TOKEN=<token>      # proxy auth key (default: freecc)
NVIDIA_NIM_API_KEY=<key>          # or whichever provider key(s) you use
```
Admin UI (`/admin`) is loopback-only — not accessible on Vercel by design.

---

## CI / Quality Gates

All enforced in `.github/workflows/tests.yml` on push/PR to `main`/`master`:

| Job | Command | Status |
|---|---|---|
| Ban type ignore suppressions | `grep -rE '# type: ignore\|# ty: ignore'` | ✅ |
| ruff-format | `uv run ruff format --check` | ✅ |
| ruff-check | `uv run ruff check` | ✅ |
| ty | `uv run ty check` | ✅ |
| pytest | `uv run pytest -v --tb=short` | ✅ 1443/1443 |

---

## Roadmap

### Completed ✅
- [x] FastAPI proxy with Anthropic-compatible routes
- [x] 17 provider backends (OpenAI compat + Anthropic native + local)
- [x] Per-model-tier routing (Opus / Sonnet / Haiku / fallback)
- [x] Streaming, tool use, thinking block support
- [x] Request optimization handlers (probe mocking, title skip, etc.)
- [x] Admin UI — config editor, provider status, validate/apply
- [x] Discord + Telegram bot wrapper with session trees
- [x] Voice note transcription (local Whisper + NVIDIA NIM Riva)
- [x] `/v1/models` Gateway model discovery
- [x] Rate limiting + concurrency control per provider
- [x] **Production Admin UI overhaul** — Toast, auto-refresh, kbd shortcuts, copy, skeleton
- [x] **Real-time settings search/filter** — topbar input, key+label match, auto-hides empty sections
- [x] **Vercel deployment** — `Dockerfile` (python:3.14-slim, non-root, uv), `vercel.json`, stderr log fallback for read-only Lambda FS
- [x] **OpenAI provider** — `providers/openai/`, `openai/gpt-4o` slug, `OPENAI_API_KEY`, wired across all 9 touch-points
- [x] **`GET /health/ready`** — authenticated readiness check: provider, model, tier overrides, auth status; useful for Vercel env var verification
- [x] **Dark/light theme toggle** — `localStorage` persistence, sun/moon SVG icons, `data-theme` on `<html>`, full light-mode design token overrides in CSS
- [x] **Per-request metrics panel** — `api/metrics.py` in-memory store, `GET /admin/api/metrics`, Admin UI "Metrics" tab: summary cards (total/avg/p95/tokens), latency sparkline, request table with inline bars
- [x] **Provider health history** — `api/health_history.py` per-provider bounded deque (50 entries), `GET /admin/api/health-history`, JSON persistence to `~/.fcc-cache/health_history.json` (serverless-safe fallback), inline sparkline on each provider card
- [x] **One-click API key validation** — `POST /admin/api/providers/{id}/validate`; `validate_credentials()` on BaseProvider; OpenAI-compat override adds 1-token completion check; "Validate key" button per non-local provider card

### Planned 🔲
- [ ] *(roadmap clear — open an issue to propose the next feature)*

---

## Changelog

### v2.0.0 — 2026-06-15 (audit + ty)
- **`ty` regression fix**: removed redundant `health_history = importlib.reload(health_history)` reassignments in `tests/api/test_admin.py` (3 sites). `importlib.reload()` mutates the module in place, so the rebinding only confused ty's narrowed-import type with the generic `ModuleType` return — no `# type: ignore` added (CLAUDE.md hard rule).
- **KafCade audit (`kc E`)**: read-only structured pass. Scores — Security 9/10, Correctness 9/10, Performance 10/10, Quality 9/10. No P0/P1 findings. Surfaced provider count drift (17 → 18, openai shipped 2026-05-30) and test count drift (1429 → 1443) in Module Map — corrected.
- **CI**: 1443/1443 passing across all 4 gates (ruff format, ruff check, ty, pytest).

### v2.0.0 — 2026-06-15 (validate-key)
- **One-click provider API key validation**: `POST /admin/api/providers/{id}/validate` (loopback-only). Calls `provider.validate_credentials(preferred_model?)`. `BaseProvider` default: model-list check. `OpenAIChatTransport` override: model list **+** 1-token `max_tokens=1` chat completion — proves the key works for actual inference.
- **Admin UI "Validate key" button**: every non-local provider card gets a dedicated "Validate key" button (distinct from "Refresh models"). Returns `{auth_ok, completion_ok, models_count, test_model?, error_type?}`. UI shows "Key valid ✓" / "Auth OK" (completion failed) / "Key invalid ✗" with toast.
- **CI**: 1443/1443 passing (9 new tests).

### v2.0.0 — 2026-06-15 (health history)
- **Provider health history**: `api/health_history.py` — per-provider `defaultdict(deque(maxlen=50))` with `threading.Lock`. `record(provider_id, status, latency_ms, error_type)` writes one outcome per `test_provider` or `_check_local_provider` call.
- **JSON persistence**: writes to `${FCC_CACHE_DIR:-~/.fcc-cache}/health_history.json` on every record; restored on module import. `_PERSIST_DISABLED` flag prevents repeat retries on read-only Vercel/Lambda FS (same pattern as `configure_logging`).
- **`GET /admin/api/health-history`** (loopback-only): returns `{providers: {<id>: [{ts, status, latency_ms, error_type?}, …]}}`.
- **Admin UI inline sparkline**: every provider card gets a `.provider-spark` strip showing last 30 health-check outcomes; bars color-coded (ok/slow/error); refreshes after every Test/Refresh-models click.
- **CI**: 1435/1435 passing (6 new tests: empty store, loopback enforcement, record+persist roundtrip, per-provider buffer cap)

### v2.0.0 — 2026-06-13 (metrics)
- **Per-request metrics panel**: `api/metrics.py` bounded deque (500 entries), thread-safe `record()`/`snapshot()`. `_metered_stream()` in `services.py` wraps provider SSE, parses `message_delta` for `output_tokens`, records latency on completion/error.
- **`GET /admin/api/metrics`**: loopback-only, returns `{requests[], summary{total, avg_latency_ms, p95_latency_ms, total_input_tokens, total_output_tokens}}`
- **Admin UI Metrics tab**: summary cards, latency sparkline (newest-right, color-coded by threshold), request table with inline latency bars. Loads on nav switch.
- **Vercel path conflict fix**: renamed `admin.css` → `fcc.css`; Vercel strips extensions before conflict checking so `admin.js` + `admin.css` → both `admin` = blocked.
- **CI**: 1431/1431 passing

### v2.0.0 — 2026-06-07 (theme + build)
- **Dark/light theme toggle**: sun/moon SVG button in topbar; `localStorage` key `fcc-theme`; `data-theme="light"` on `<html>`; full CSS variable override block for light mode; persists across page loads
- **Hatchling wheel fix**: removed duplicate `api/admin_static` from `force-include` — newer hatchling strict-mode rejected the double-inclusion; `uv build` now clean; unblocked Vercel re-deployment
- **CI**: 1429/1429 passing; deployed to https://free-claude-code-main-ebon.vercel.app

### v2.0.0 — 2026-05-30 (readiness)
- **`GET /health/ready`** (auth required): returns `{status, provider, model, model_opus, model_sonnet, model_haiku, auth_required}` — verify Vercel env wiring without local admin UI

### v2.0.0 — 2026-05-30 (providers)
- **OpenAI provider** (`openai/`): `providers/openai/OpenAIProvider` extends `OpenAIChatTransport`, base `https://api.openai.com/v1`, thinking via `ReasoningReplayMode.REASONING_CONTENT` for o1/o3/o4 models
- `OPENAI_API_KEY` + `OPENAI_PROXY` wired through settings, admin config, `.env.example`, admin UI, catalog, registry, and test mocks (18th provider)
- Smoke config updated: `openai/gpt-oss-120b:free` → `open_router/openai/gpt-oss-120b:free` to avoid provider ID clash
- **DeepSeek** already supported — `MODEL=deepseek/deepseek-chat` + `DEEPSEEK_API_KEY` (no code change needed)

### v2.0.0 — 2026-05-30 (deploy)
- **Live on Vercel**: https://free-claude-code-main-ebon.vercel.app — Python 3.14, uv 0.10.11, iad1
- **Dockerfile**: multi-stage `python:3.14-slim-bookworm`, non-root `fcc` user, `${PORT:-8082}`, compatible with Vercel/Railway/Render
- **`configure_logging` serverless fix**: `PermissionError/OSError` fallback to stderr when Lambda filesystem is read-only
- **Removed `[tool.uv] required-version`** from `pyproject.toml` — blocked Vercel's uv 0.10.11; `uv.lock` provides reproducibility

### v2.0.0 — 2026-05-30
- **Real-time settings search/filter**: topbar `<input type="search">`, filters all `.field` elements live by env-var key + human label; empty section cards auto-hide; clears on view-switch

### v2.0.0 — 2026-05-29
- **Production Admin UI overhaul**: Toast system, 30 s auto-refresh with pulsing dot, `Ctrl+S` / `Ctrl+Enter` shortcuts, copy-to-clipboard, skeleton loading, unsaved-changes guard, 6 CSS keyframe animations, provider card lift-on-hover, accent glow on focused inputs, full mobile reflow
- Fixed `.gitignore` `server.*` wildcard over-matching `server.py` — added `!server.py` negation
- 1429 / 1429 tests passing

### v2.0.0 (upstream baseline)
- 17 provider backends
- Messaging platforms (Discord, Telegram), voice transcription
- Admin UI for managed `.env` config
- Gateway model discovery via `/v1/models`
- Python 3.14, uv, ruff, ty CI stack
