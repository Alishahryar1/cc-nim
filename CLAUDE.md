# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Keep AGENTS.md and CLAUDE.md identical: apply every change to this file to both.

## Project Overview

Free Claude Code (FCC) is a local proxy that connects coding agents to AI providers. It accepts Anthropic Messages traffic from Claude Code and Pi, and OpenAI Responses traffic from Codex, routes each request to a configured upstream provider (30+ hosted APIs plus local Ollama / LM Studio / llama.cpp), and preserves the caller's wire protocol.

Three runtime surfaces:

- **HTTP proxy** — FastAPI app in `src/free_claude_code/api/`: `/v1/messages`, `/v1/responses`, `/v1/messages/count_tokens`, `/v1/models`, `/health`, `/stop`, plus the local-only Admin UI at `http://127.0.0.1:8082/admin`.
- **CLI launchers** — `fcc-server`, `fcc-claude`, `fcc-codex`, `fcc-pi`, `fcc-desktop` (entrypoints declared in `pyproject.toml`).
- **Messaging bridge** — optional Discord/Telegram adapters that turn chat messages into managed Claude CLI sessions.

## Commands

Everything runs through [uv](https://docs.astral.sh/uv/) with Python 3.14 — never a global interpreter. Requires uv >= 0.11.16 (`[tool.uv] required-version` in pyproject.toml); install or update uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` and run `uv python install 3.14.0` if that interpreter is missing. See `.env.example` for environment variables.

```bash
uv run fcc-server                  # run the proxy + Admin UI from a checkout

./scripts/ci.sh                    # full local CI before pushing (Windows: .\scripts\ci.ps1)
./scripts/ci.sh --only pytest      # subset; also --skip, --dry-run (PowerShell: -Only, -Skip, -DryRun)
```

`scripts/ci.sh` runs, in order: suppression grep → `uv run ruff format` → `uv run ruff check --fix` → `uv run ty check` → `uv run pytest -v --tb=short`. Ruff runs in repair mode locally; GitHub CI is check-only (`ruff format --check`, `ruff check`) and enforces the same 5 check IDs as parallel jobs on push, pull_request, and merge_group. Use the individual repair commands above when debugging local failures; use the `--check` variants only when verifying GitHub-style enforcement.

```bash
uv run pytest                                             # deterministic tests; addopts "-n auto" (parallel xdist)
uv run pytest tests/core/test_failures.py -v              # one file
uv run pytest tests/core/test_failures.py::test_name -v   # one test (add -n 0 to serialize while debugging)
```

`tests/` mirrors the `src/` package layout and must stay hermetic. Live product smoke lives in `smoke/` and does nothing without opt-in (target catalog, env vars, and failure classes are in `smoke/README.md`):

```bash
FCC_LIVE_SMOKE=1 uv run pytest smoke -n 0 -s --tb=short
FCC_LIVE_SMOKE=1 FCC_SMOKE_TARGETS=api,providers uv run pytest smoke -n 0 -s --tb=short
```

## Architecture

[ARCHITECTURE.md](ARCHITECTURE.md) is the authoritative map — read it before changing package boundaries, providers, protocol conversion, launchers, or messaging. It also has step-by-step extension checklists (add a provider, admin setting, messaging platform, or protocol behavior). The essentials:

**Import policy (test-enforced).** Production code lives under `src/free_claude_code/`. An AST-scanner contract test derives every static production edge and enforces this exact direct-dependency matrix; the first-party module graph must stay acyclic. New packages or cross-package edges are added to the policy deliberately:

| Package | Owns | May import |
| --- | --- | --- |
| `config` | pydantic-settings schema, provider catalog, paths, Admin config | — |
| `core` | protocol-neutral logic: Anthropic + OpenAI Responses wire models and conversion, SSE, canonical `ExecutionFailure` semantics, credential-redacting diagnostics, token counting | — |
| `application` | dependency leaf: `ModelRouter`, `ProviderExecutor`, reasoning resolution, consumer-owned ports | config, core |
| `providers` | provider runtime/construction, OpenAI-chat profiles, specialized adapters, SDK/HTTP failure classification, retry/recovery, admission | application, config, core |
| `api` | FastAPI app, routes, product handlers, local optimizations, web tools, model-catalog responses | application, config, core |
| `messaging` | Discord/Telegram runtimes, tree queues, transcripts, persistence, voice flow | core |
| `cli` | entrypoints, client launchers, managed Claude subprocesses | config, core |
| `runtime` | process composition root: bootstrap, provider generations, ordered lifecycle shutdown | everything |

The single sanctioned exception: `cli.entrypoints` → `runtime.bootstrap`.

**Request flow.** Route → provider-generation lease (`runtime/provider_manager.py`) → product handler (`api/handlers/`) → `ModelRouter` resolves model + reasoning intent → `ProviderExecutor` (`application/execution.py`) preflights and streams → provider adapter → Anthropic SSE. Codex Responses traffic goes through `core/openai_responses/`, which converts Responses ⇄ Anthropic Messages at the adapter boundary. Failures cross boundaries as canonical `core/failures.py` values; each protocol package maps them to its wire error type, and the HTTP commit boundary (first chunk sent) decides between a non-2xx JSON error and a terminal stream event.

**Provider model.** Neutral provider metadata is centralized in `config/provider_catalog.py`. Most upstreams are declarative `OpenAIChatProfile` data on the shared `OpenAIChatProvider` (`providers/openai_chat/`) — configuration differences stay data, not subclasses. Only a true upstream quirk (own state, model-list behavior, stream events, retry algorithms) justifies a specialized package plus sparse factory entry, and the union of construction owners must exactly equal the neutral catalog.

**Configuration.** Flat schema in `config/settings.py`. Dotenv precedence: repo `.env` → `~/.fcc/.env` → optional `FCC_ENV_FILE` (later files override earlier ones). `.env.example` is the Admin UI template source (force-included into the wheel), not a live config file. Model refs are `<provider_id>/<model-id>`; `MODEL` is the fallback and `MODEL_FABLE` / `MODEL_OPUS` / `MODEL_SONNET` / `MODEL_HAIKU` are Claude-tier overrides. Reasoning is resolved once at the application boundary into an immutable `core/reasoning.py` policy; provider adapters translate only the subset their documented wire API can express, and never branch on upstream model names or versions.

**Stable contract.** The customer-facing surfaces are the contract: `fcc-server`/Admin UI, `fcc-claude`/Claude proxy behavior, `fcc-codex`/Responses behavior, `fcc-pi`, the messaging bridges, and the install scripts. Internal modules, class designs, helper APIs, and tests are not stable — refactor them freely when it simplifies the system, and update tests that encode obsolete internal shapes to assert customer-facing behavior instead.

## Invariants (CI-enforced — never suppress)

- Never add `# type: ignore`, `# ty: ignore`, or `from __future__ import annotations` — a CI grep fails on all three. Fix the underlying type issue or import cycle instead (move shared types/protocols to a neutral owner).
- Python 3.14 native lazy annotations are the standard; the py314 ruff target allows multiple exception types in `except` without parentheses (`except TypeError, ValueError:`).
- Prefer top-level imports; avoid `TYPE_CHECKING` and local imports for first-party or required dependencies.
- Optional voice imports are lazy at their exact owners, below a function boundary: `torch` / `transformers` / `librosa` in `messaging.transcription`; `riva.client` in `providers.nvidia_nim.voice`.
- Shared Anthropic protocol logic belongs in `core/anthropic/`; no provider imports from another provider.
- Provider-specific config fields (e.g. `nim_settings`) stay in provider constructors, not the base `ProviderConfig`.

## Engineering Principles

- Zero-defect, root-cause-oriented engineering for bugs; test-driven engineering for new features. Write the simplest code possible; keep the codebase minimal and modular.
- DRY: extract shared base classes to eliminate duplication; prefer composition over copy-paste.
- Encapsulation: use accessor methods for internal state (e.g. `set_current_task()`), not direct `_attribute` assignment from outside.
- Dead code: remove unused code, legacy systems, and hardcoded values; use settings/config instead of literals (e.g. `settings.provider_type`, not `"nvidia_nim"`).
- Performance: list accumulation for strings (not `+=` in loops), cache env vars at init, prefer iterative over recursive when stack depth matters.
- Platform-agnostic naming in shared code (e.g. `PLATFORM_EDIT`, not `TELEGRAM_EDIT`).
- Complete migrations: when moving modules, update imports to the new owner and remove old compatibility shims in the same change unless preserving a published interface is explicitly required.
- Maximum test coverage for everything, preferably including live smoke coverage to catch bugs early.

## Workflow

1. **ANALYZE**: Read relevant files. Do not guess.
2. **PLAN**: Map out the logic. Identify root cause or required changes. Order changes by dependency.
3. **EXECUTE**: Fix the cause, not the symptom. Execute incrementally with clear commits. Do exactly as much as asked; nothing more, nothing less. Changes impact multiple files — propagate updates correctly.
4. **VERIFY**: Run `./scripts/ci.sh` (or `.\scripts\ci.ps1`), plus relevant smoke tests when needed. Confirm the fix via logs or output.
5. **VERSION**: If the commit touches production files on `main`, bump semver in the same commit (see below).

## Versioning (MAIN)

Every commit on `main` that changes a **production file** must include a semver bump in **`pyproject.toml`** in the **same commit**. Do not merge or push prod changes without updating the version.

**Production files** (runtime, packaging, or install surface):

- `src/free_claude_code/api/`, `application/`, `cli/`, `config/`, `core/`, `messaging/`, `providers/`
- `.env.example`
- `pyproject.toml` (dependencies, scripts, packaging)
- `scripts/install.sh`, `scripts/install.ps1`, `scripts/uninstall.sh`, `scripts/uninstall.ps1`, `scripts/ci.sh`, `scripts/ci.ps1`

These do **not** require a version bump on their own:

- `tests/`, `smoke/`
- Docs and assets: `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `assets/`, `AGENTS.md`, `CLAUDE.md`
- CI and repo config: `.github/`, `.gitignore`

If a single commit mixes production and non-production edits, still bump the version.

**Semver rules** (`MAJOR.MINOR.PATCH` in `[project].version`):

- **PATCH** (`x.y.Z+1`): bug fixes, refactors with no user-visible behavior change, dependency updates, packaging/install fixes.
- **MINOR** (`x.Y+1.0`): backward-compatible features — new providers, admin fields, CLI commands, config options, or behavior additions.
- **MAJOR** (`X+1.0.0`): breaking changes — removed or renamed env vars, incompatible API/CLI/default changes, or migrations users must act on.

When unsure between PATCH and MINOR, prefer PATCH for fixes and MINOR for new capability.

**Required steps**: classify the change → update `version` in `pyproject.toml` → run `uv lock` so `uv.lock` reflects the new package version → include the version and lockfile updates in the same commit as the production change. Example: after a packaging fix, bump `1.2.38` → `1.2.39`, run `uv lock`, commit together with the fix.

## CI Notes

- Required status checks are exactly the 5 check IDs (possibly prefixed with `CI /`): **Ban suppressions and legacy annotations**, **ruff-format**, **ruff-check**, **ty**, **pytest**. Remove **ci** from required checks if it was previously added for the old gate job.
- Repository protection uses rulesets: a non-bypassable main integrity ruleset requires pull requests, merge queue, and required checks, and blocks direct/force pushes to `main`; a separate review ruleset may allow `Alishahryar1`/admins to bypass review only.

## Summary Standards

Summaries must be technical and granular. Include: [Files Changed], [Logic Altered], [Verification Method], [Residual Risks] (if no residual risks then say none).
