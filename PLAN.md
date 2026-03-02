# cc-nim Homebrew Integration & CLI Implementation Plan

## Objective
Add Homebrew formula and CLI commands (`init`, `start`, `stop`) to manage cc-nim server lifecycle and configuration on macOS/Linux.

## Scope
- Add `typer` as a dependency for consistent CLI exit handling.
- Implement CLI command functions in `cc_nim/cli.py` with proper error reporting and exit codes.
- Manage configuration via `~/.ccenv` with `CCPROXY_CONFIG` override.
- Use PID file `~/.cc-nim/cc-nim.pid` for start/stop coordination.
- Provide `brew services` integration for stop command (fallback to PID kill).
- Write comprehensive unit tests covering all commands and edge cases.
- Update documentation (`README.md`) and `run.sh` (deprecation notice).
- Create Homebrew formula `Formula/cc-proxy.rb`.

## Implementation Notes
- **Config Path Resolution**: Priority order: `CCPROXY_CONFIG` > `~/.ccenv` > `cwd/.env`. `_get_config_path()` handles this.
- **PID Handling**: `start_command` checks for existing PID; if process is running, exits with error; if stale, removes PID and continues. Writes current PID on start; `stop_command` removes PID on exit.
- **Error Handling**: CLI commands raise `typer.Exit(code)` on failures to match test expectations.
- **Testing**: Tests located under `tests/cli/` use `pytest` with monkeypatching for isolation.

## Task List

| ID | Task | Status | Notes |
|----|------|--------|-------|
| 1 | Fork repository to `rainbow` org and create `feature/homebrew-setup` branch. | ⏳ Pending | |
| 2 | Update `pyproject.toml`: add `typer` dependency; ensure `[project.scripts]` entry point. | ✅ Completed | `typer>=0.9.0` added |
| 3 | Implement `init_command` (create config template). | ✅ Completed | |
| 4 | Implement `start_command` with PID check and `uvicorn.run`. | ✅ Completed | Includes stale PID handling |
| 5 | Implement `stop_command` with brew service first, PID fallback. | ✅ Completed | Brew stop tried; PID kill if needed |
| 6 | Write unit tests for CLI commands (`tests/cli/test_cli_commands.py`). | ✅ Completed | Covers success and error paths |
| 7 | Fix async test fixtures (`tests/conftest.py`) to use `AsyncMock`. | ✅ Completed | Resolved `MagicMock` await errors |
| 8 | Update `README.md` with Homebrew installation and usage. | ⏳ Pending | |
| 9 | Create Homebrew formula `Formula/cc-proxy.rb`. | ⏳ Pending | |
| 10 | Update `run.sh` to show deprecation notice and direct to `cc-nim`. | ⏳ Pending | |
| 11 | Run full test suite (`uv run pytest`) and fix any failures. | 🚧 In Progress | Code changes complete; tests not yet executed |
| 12 | Run lint/type checks (`ruff format`, `ruff check`, `ty`). | ⏳ Pending | Will run after tests |
| 13 | Commit changes on `feature/homebrew-setup` and push to fork. | ⏳ Pending | |
| 14 | Create pull request to upstream repository. | ⏳ Pending | |

## Completed Code Changes
- Added `import typer` to `cc_nim/cli.py`.
- Changed all CLI error returns to `raise typer.Exit(1)`.
- Added PID existence and liveness check in `start_command`.
- Modified tests to expect `typer.Exit` and fixed stale PID mock to use `ProcessLookupError`.
- Updated `tests/conftest.py` to use `AsyncMock` for async methods.

## Next Steps
1. Run `uv sync` to ensure `typer` installed in environment.
2. Execute `uv run pytest` to verify all tests pass.
3. Run `uv run ruff format` and `uv run ruff check`.
4. Run `uv run ty check`.
5. If any failures, debug and fix.
6. After all checks pass, commit and push to remote.
7. Update `README.md`, create formula, and update `run.sh`.
8. Open pull request.

---
*Last updated: 2026-03-02*
