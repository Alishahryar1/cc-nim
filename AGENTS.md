# FCC Development Instructions

Keep `AGENTS.md` and `CLAUDE.md` byte-for-byte identical.

Free Claude Code is a local Python gateway connecting coding agents to configured
AI providers. Read [ARCHITECTURE.md](ARCHITECTURE.md) for ownership and request
flows, [README.md](README.md) for user workflows, and [.env.example](.env.example)
for documented settings.

## Tooling

- Commands assume the repository root. Unrelated edits and untracked files are
  outside the change scope; use `git ls-files` for tracked paths and `rg` for search.
- Use `uv` for Python environments and `uv run` for Python tools/scripts. Follow
  [.python-version](.python-version), `[tool.uv]` in [pyproject.toml](pyproject.toml),
  and [uv.lock](uv.lock). Install missing prerequisites as needed; do not upgrade
  tools or dependencies during unrelated work.
- `uv sync --locked` prepares the development environment. Optional voice extras
  are needed only for the corresponding live transcription work.
- Never use the real `~/.fcc/` configuration, credentials, chat database, or
  messaging state as disposable test fixtures. Use isolated temporary state.

| Task | Command |
| --- | --- |
| Focused deterministic tests | `uv run pytest tests/<area>/test_<feature>.py -n 0 -q` |
| Format / lint repair | `uv run ruff format <changed-paths>`; `uv run ruff check --fix <changed-paths>` |
| Read-only format / lint checks | `uv run ruff format --check`; `uv run ruff check` |
| Type checking | `uv run ty check` |
| Full deterministic tests | `uv run pytest -v --tb=short` |
| Browser setup / tests | `uv run playwright install chromium`; `uv run pytest e2e -n 0 -v --tb=short` |
| Full local CI | Windows: `.\scripts\ci.ps1`; macOS/Linux: `./scripts/ci.sh` |

Use full local CI before pushing code. The scripts run Ruff in repair mode
before type checking and tests; review their diff. Use PowerShell
`-Only` / `-Skip` / `-DryRun` or shell `--only` /
`--skip` / `--dry-run` when iterating. Check IDs are `suppressions`,
`ruff-format`, `ruff-check`, `ty`, `pytest`, and `playwright`.

## Ownership And Contracts

- `runtime/` composes services and owns process lifecycle and provider generations.
  `api/` owns HTTP validation, authentication, product handlers, and response
  commitment. `application/` owns routing, reasoning intent, model fallback, and
  Chat use cases. Keep these responsibilities separate.
- `core/` owns shared protocol models/conversion, failure semantics, and redaction.
  `providers/` alone classifies SDK/HTTP failures and owns upstream attempts,
  retries, and recovery. Protocol/API boundaries choose wire errors.
- Provider metadata belongs in `config/provider_catalog.py`; ordinary upstream
  differences belong in OpenAI Chat profiles. Add a specialized adapter only for
  behavior/state a profile cannot express. Keep provider-only settings out of
  shared `ProviderConfig`.
- Resolve reasoning intent at the application boundary. Providers encode supported
  controls; never infer reasoning behavior from upstream model names/versions.
  Keep reasoning replay distinct from controls for the next generation.
- Preserve each ingress protocol through the application boundary and the four
  ingress/upstream cells. Do not introduce a lossy universal request model.
  Preserve tool identity, call/result pairing, images, reasoning, usage, and
  event order; reject unrepresentable input instead of silently dropping it.
- Preserve stream commitment and resource ownership: no model fallback after
  output commits, no duplicate terminal events, and no lease release before its
  response/stream closes. Cancellation must drain owned work; shutdown must not
  abandon partially closed resources or overlap replacement runtimes.
- Keep config loading/provenance in `config/loader.py`, sparse Admin persistence
  in `config/admin/`, and runtime Apply in `runtime/application.py`. The managed
  `~/.fcc/.env` is the live file; repository `.env.example` is documentation.
- Keep Chat revisions, operation settlement, and durable state consistent.
  Messaging identities include platform/chat scope; stale tasks must not revive
  cleared branches. Launchers own temporary client overlays, not native user
  credentials/configuration.
- Follow the exact dependency/facade policy in
  [test_import_boundaries.py](tests/contracts/test_import_boundaries.py).
  Change allowed edges deliberately, never just to silence a failing contract.

## Python And Design

- Use Python 3.14 native annotations. Do not add
  `from __future__ import annotations`, `# type: ignore`, or `# ty: ignore`.
- Use owner-defined types; `JsonValue`/`JsonObject` for JSON; narrowed `object`
  only at opaque boundaries. Avoid `typing.Any`; fix the underlying model.
- Prefer top-level imports. Resolve cycles through ownership, not
  `TYPE_CHECKING` or annotation stringization. Preserve deliberate lazy loading
  at existing optional-dependency, provider-factory, and lightweight CLI owners.
- Prefer simple composition and shared helpers over duplicated behavior or
  unnecessary base classes. Use public operations rather than another object's
  private mutable state; keep platform/provider quirks at their adapters.

## Verification By Change

Changed behavior needs regression coverage, including relevant failure paths.

| Change | Required evidence |
| --- | --- |
| Protocol/provider | Relevant core/provider tests; matrix/stream contracts for conversion or event changes |
| Routing/lifecycle/failure | Application/API/runtime tests, including fallback, disconnect, cancellation, and cleanup paths affected |
| Settings/Admin | Config/API contracts; deterministic `e2e/` tests for changed browser interactions |
| Chat/messaging | Persistence, revision/scope isolation, concurrency, and stop/clear tests affected |
| CLI/install/packaging | Launcher/script/packaging contracts; relevant installed-client smoke when needed |
| Documentation only | Source accuracy, local links, instruction-file equality, and scoped `git diff --check` |

Live checks are under [smoke/](smoke/README.md), separate from deterministic CI.
Use `FCC_LIVE_SMOKE=1` and targeted `FCC_SMOKE_TARGETS` only for the relevant
configured services; prerequisites alone do not prove product behavior. Live
checks can consume quota or send/delete bot messages: use only authorized
targets and report skips or missing prerequisites accurately.

GitHub's [CI workflow](.github/workflows/tests.yml) runs Ruff in check-only mode
and enforces all six checks; failing required checks block merge. The
dependency-cache workflow is not a validation substitute.

## Versioning And Documentation

Every commit merged to `main` that changes production files must include a semver
bump in `pyproject.toml` and the matching
`uv lock` update. Production includes all installed code/assets under
`src/free_claude_code/` (including `runtime/`), `.env.example`, runtime/dependency/
packaging metadata, and install/uninstall/CI scripts.

Use **PATCH** for fixes/refactors/dependency or packaging updates, **MINOR** for
backward-compatible capabilities, and **MAJOR** for breaking API/CLI/configuration
changes. Docs, tests, smoke tests, and repository workflow changes alone do not
require a bump.

Update `ARCHITECTURE.md` when ownership, public protocols, lifecycle, configuration,
or extension paths change; update `README.md` and `.env.example` for user-facing
setup/configuration changes. Keep volatile defaults and catalogs in their source
owners, and keep these two instruction files concise and identical.
