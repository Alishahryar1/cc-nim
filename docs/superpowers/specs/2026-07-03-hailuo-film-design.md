# `/hailuo-film` — Seedance-recipe film generation via Hailuo (browser-driven)

**Date:** 2026-07-03
**Status:** Approved (design phase)
**Supersedes / adapts:** Higgsfield Seedance 4K breakdown recipe — substituted to Hailuo (MiniMax) and driven through the user's existing FCC-proxy + skills stack.

## Goal

Produce high-quality, cross-shot-consistent AI short films on the user's own infrastructure by adapting the Higgsfield Seedance 4K workflow to Hailuo. The FCC-proxied text model acts as the **prompt architect**; Hailuo's web UI (driven via Chrome DevTools MCP, user signs in once) acts as the **renderer**. Output is a project folder of generated asset images and video clips plus a handoff manifest, ready for assembly by the existing `reel`/`hyperframes` skills (out of scope here).

## Non-goals (YAGNI)

- No final assembly — `reel` (FFmpeg NLE) and `hyperframes` (HTML→video) already own that step.
- No Hailuo/MiniMax official API integration — browser automation only (user's choice).
- No auto-login or credential storage — user signs in interactively; the skill never touches credentials.
- No headless/cron mode — interactive-only (browser-driven, user-signed-in session).
- No character LoRA / identity-lock — Hailuo's web tier doesn't expose it; revisit if an API path is added later.

## Architecture

A 3-phase pipeline. A project folder on disk is the single source of truth; a `progress.json` makes every run resumable.

```
~/.fcc proxy (text model)            Chrome DevTools MCP (real Chrome, user signs in)
        │                                              │
  Phase 1: Architect                             Phases 2 & 3: Drive Hailuo
  concept ──► architect.py ──► shots.json        assets/ + clips/
        │                         │                      │
        │                  asset manifest ───────────────┘
        │                         │                      │
        └──────────────── project/progress.json (resumable state) ────┘
                                  │
                          manifest.md (handoff → reel/hyperframes)
```

**Project folder** (default `~/hailuo-projects/<slug>-<timestamp>/`):
- `shots.json` — shot list + per-shot 16-section prompt + asset manifest
- `assets/<asset-id>.png` — generated reference images
- `clips/<shot-id>.mp4` — generated video clips
- `progress.json` — per-asset and per-shot status: `pending | done | failed`
- `manifest.md` — human-readable handoff doc for assembly

## Components

| File | Purpose |
|------|---------|
| `.claude/skills/hailuo-film/SKILL.md` | Skill entry. Defines `/hailuo-film <concept>`, the 3-phase workflow, the sign-in gate, and embeds the full 16-section recipe (drawn from `reference/seedance-recipe.md`). |
| `.claude/skills/hailuo-film/reference/seedance-recipe.md` | Canonical source of the 16-section prompt template + asset rules. One place to edit the recipe; SKILL.md and `architect.py` both consume it. |
| `.claude/skills/hailuo-film/scripts/architect.py` | Concept → `shots.json` + asset manifest. Calls FCC proxy `http://localhost:8082/v1/messages` with `ANTHROPIC_AUTH_TOKEN=freecc`, using the session's current model (or `MODEL` override). Emits structured JSON. Fails fast if proxy is down. |
| `.claude/skills/hailuo-film/scripts/hailuo_driver.md` | Browser playbook Claude reads at runtime — step-by-step for Hailuo image-gen and video-gen (I2V): open URL, wait for sign-in, upload refs, paste prompt, generate, wait for render, download. |
| `.claude/skills/hailuo-film/scripts/selectors.json` | UI selector map (text labels / a11y roles). The single maintenance surface when Hailuo changes their UI. |
| `.claude/skills/hailuo-film/scripts/project.py` | Project scaffolding + `progress.json` resume helpers (mark done/failed, next-pending lookup, skip-done logic). |
| `.claude/skills/hailuo-film/tests/test_architect.py` | Unit test: architect prompt-building + JSON schema validation against a mocked proxy (no browser). |
| `.claude/skills/hailuo-film/tests/test_project.py` | Unit test: resume logic — done items skipped, failed retried (or skipped per flag). |

## The 16-section prompt (per shot)

Adapted verbatim from the Seedance 4K breakdown; model-agnostic. The architect emits one JSON object per shot with these keys:

1. `SCENE_CONTEXT`
2. `ACTIVE_REFERENCES`
3. `LOCATION_MAP`
4. `FIRST_FRAME_BLOCKING`
5. `FORMAT_MODE`
6. `OPTICS`
7. `CAMERA`
8. `ACTION`
9. `PERFORMANCE`
10. `PHYSICS`
11. `LIGHTING`
12. `COLOR_GRADE`
13. `AUDIO`
14. `STYLE`
15. `OUTPUT_SETTINGS`
16. `POSITIVE_LOCKS`

Plus a top-level `asset_manifest` listing recurring entities (character sheet, hero locations, props) to generate as standalone images before any shot.

## Data flow

1. User invokes `/hailuo-film <concept>` (concept is a string or path to a concept file).
2. `architect.py` runs against the FCC proxy → writes `shots.json` + asset manifest into a new project folder.
3. Claude reads `hailuo_driver.md`, opens Hailuo in Chrome via Chrome DevTools MCP `new_page`.
4. **Sign-in gate:** the skill asks the user in chat to confirm they've signed in. It never auto-logs in and never stores credentials. Once confirmed, it proceeds.
5. **Asset phase:** for each asset in the manifest — drive Hailuo image-gen, upload any reference images, paste the asset prompt, generate, wait, download to `assets/<id>.png`, mark `progress.json`.
6. **Shot phase:** for each shot — two sub-steps:
   - **6a. First-frame:** drive Hailuo image-gen with the shot's `FIRST_FRAME_BLOCKING` prompt, using the shot's referenced library assets as input references, to produce the shot's composition frame; download to `assets/<shot-id>-firstframe.png`.
   - **6b. Animate:** drive Hailuo video-gen (image-to-video) on that first-frame, paste the full 16-section prompt, generate, wait, download to `clips/<id>.mp4`, mark `progress.json`.
7. Write `manifest.md` (shot order, asset refs, clip paths) as the handoff doc for `reel`/`hyperframes`.
8. On any re-run: skip `done` items, retry `failed` (or skip per a flag), continue from the next `pending` item.

## Error handling

- **Selector drift** (Hailuo UI changes): a drive step cannot find an element → halt loudly with a "update `selectors.json`" message. Never mis-click. `progress.json` preserves all completed work.
- **Generation failure** (NSFW filter / rate limit / render error): log the shot as `failed` in `progress.json` with the error, continue to the next shot, report all failures at the end.
- **Browser session lost / page closed:** on resume, re-open, re-navigate, continue from `progress.json`'s next pending item.
- **FCC proxy not running:** `architect.py` fails fast with a clear `start the proxy: fcc` message.
- **Long films:** architect warns if shot count is high and suggests splitting the concept.

## Testing

- `test_architect.py` — offline unit test: architect prompt construction + `shots.json` schema validation (16 sections present, asset manifest well-formed) against a mocked `/v1/messages` response. No browser.
- `test_project.py` — offline unit test: `progress.json` resume logic — `done` skipped, `failed` retried or skipped per flag.
- **Smoke test** (manual, documented in SKILL.md): a 1-shot, 1-asset concept ("a single shot of a cat on a couch, 5 seconds") run end-to-end against the real Hailuo UI after the user signs in. Validates the browser layer. Re-run this whenever Hailuo's UI changes.
- No automated tests for the browser layer (inherently flaky against a live third-party site). `selectors.json` + `hailuo_driver.md` are the documented maintenance surface.

## Sign-in / security

- The skill opens Hailuo in a real Chrome via Chrome DevTools MCP and explicitly waits for the user to confirm sign-in in chat.
- The skill never enters, reads back, or stores any credential. No passwords, no cookies exported, no token capture.
- All generated assets/clips are local files in the project folder; nothing is uploaded anywhere by the skill itself.

## Open questions / assumptions

- **Hailuo product surface assumed:** image generation (for assets + first-frames) and image-to-video generation (for shot clips), as exposed on the Hailuo web UI. Exact on-page control names will be captured in `selectors.json` during the smoke test; the design assumes the playbook, not specific selectors.
- **Brain model:** defaults to the session's current FCC-routed model (opus-tier, e.g. `glm-5.2:cloud`). Overridable via `MODEL` env var when invoking `architect.py`.
- **Project folder location:** default `~/hailuo-projects/`; overridable.