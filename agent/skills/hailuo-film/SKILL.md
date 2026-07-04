---
name: hailuo-film
description: "Generate short films / shot lists from a concept, drive Hailuo image-gen and I2V video-gen in Chrome, download assets/clips, and hand off a manifest for reel/hyperframes assembly"
---

# `/hailuo-film` — Hailuo-driven short-film generator

Turn a one-line concept into a structured shot list, generate reference assets and I2V clips in Hailuo, and produce a manifest for downstream assembly (reel/hyperframes).

## When to invoke

- User says: "make me a Hailuo film about ...", "generate a short film", "storyboard this concept for Hailuo", "/hailuo-film <concept>", or mentions Hailuo video generation for multiple shots/assets.
- The task involves: shot list → reference images → image-to-video clips → download → manifest.

## What it does (3-phase workflow)

1. **Architect** — concept → `shots.json` + asset manifest using the 16-section Seedance recipe.
2. **Asset generation** — drive Hailuo image-gen to create recurring characters, locations, props; download to `assets/<id>.png`.
3. **Shot generation** — drive Hailuo I2V using each shot's first-frame + 16-section prompt; download to `clips/<id>.mp4`.

## Sign-in gate

Hailuo requires a signed-in browser session. This skill **never auto-logs in and never stores credentials**.

1. Open `https://hailuoai.video/` in Chrome (via Chrome DevTools MCP `new_page`).
2. Ask the user: "Please sign in to Hailuo in the browser, then reply 'done'."
3. Wait for confirmation before generating.

## Usage

```bash
# From the project root (or any directory)
/hailuo-film "A 5-shot product-launch teaser for a rugged AI drone, cinematic, desert sunrise"
```

## Project folder

Default: `~/hailuo-projects/<slug>-<timestamp>/`

```
<project>/
├── SKILL.md (this file, copied in)
├── reference/seedance-recipe.md
├── shots.json              — shot list + per-shot 16-section prompts + asset manifest
├── assets/                 — generated reference images
├── clips/                  — generated I2V clips
├── progress.json           — resume state (done/failed/pending)
└── manifest.md             — handoff doc for reel/hyperframes
```

## Components

| File | Purpose |
|------|---------|
| `scripts/architect.py` | Concept → `shots.json` via FCC proxy + Seedance recipe |
| `scripts/hailuo_driver.md` | Browser playbook Claude reads at runtime |
| `scripts/selectors.json` | UI selector map |
| `scripts/project.py` | Scaffolding + `progress.json` helpers |
| `tests/test_architect.py` | Mocked architect tests |
| `tests/test_project.py` | Resume-logic tests |
| `reference/seedance-recipe.md` | Canonical 16-section prompt template |

## Environment

- FCC proxy: `http://localhost:8082/v1/messages`
- Auth token: `ANTHROPIC_AUTH_TOKEN=freecc`
- Model routing: the FCC proxy remaps Claude tiers via `~/.fcc/.env` `MODEL_*`.
  The OPUS tier routes to `glm-5.2:cloud`, a heavy reasoner that burns the whole
  token budget thinking and emits no text — **do not use OPUS for the architect**.
  Use the HAIKU tier instead, which routes to `deepseek-v4-flash:cloud` (emits
  text directly): `HAILUO_MODEL=claude-haiku-4-5 HAILUO_MAX_TOKENS=8192`.
- The proxy returns a hybrid body: a complete JSON envelope followed by a
  trailing SSE diagnostic stream (often "Provider stream ended without
  message_stop."). `architect._extract_envelope_text` takes the envelope's text
  block and ignores the SSE tail; pure-SSE responses are handled as a fallback.
- Requires: `python3` (stdlib `urllib` only — no third-party deps).

## Error handling

- **Selector drift** → halt loudly with "update selectors.json".
- **Generation failure** (NSFW filter / rate limit / render error) → log shot as failed, continue to next, report at end.
- **Browser session lost** → re-open, re-navigate, resume from `progress.json`.
- **FCC proxy down** → architect fails fast with `start the proxy: fcc`.

## Scope (YAGNI)

- No assembly (reel/hyperframes own it).
- No API integration (browser only).
- No auto-login / credential storage.
- No headless / cron (interactive only).
- No character LoRA locking (Hailuo free tier doesn't support it).

## Testing

- Offline: `python -m pytest tests/`
- Smoke: 1-shot / 1-asset concept run end-to-end after signing in.
