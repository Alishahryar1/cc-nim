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
| 8 | Update `README.md` with Homebrew installation and usage. | ✅ Completed | Homebrew install and service usage documented |
| 9 | Create Homebrew formula `Formula/cc-proxy.rb`. | ✅ Completed | Formula added with service + post_install init |
| 10 | Update `run.sh` to show deprecation notice and direct to `cc-nim`. | ✅ Completed | Deprecation note present |
| 11 | Run full test suite (`uv run pytest`) and fix any failures. | ✅ Completed | 864 passed |
| 12 | Run lint/type checks (`ruff format`, `ruff check`, `ty`). | ✅ Completed | All checks passing |
| 13 | Commit changes on `feature/homebrew-setup` and push to fork. | ⏳ Pending | |
| 14 | Create pull request to upstream repository. | ⏳ Pending | |

## Completed Code Changes
- Added `import typer` to `cc_nim/cli.py`.
- Changed all CLI error returns to `raise typer.Exit(1)`.
- Added PID existence and liveness check in `start_command`.
- Modified tests to expect `typer.Exit` and fixed stale PID mock to use `ProcessLookupError`.
- Updated `tests/conftest.py` to use `AsyncMock` for async methods.

## Next Steps
1. Commit all completed Homebrew/CLI/test-fix changes on `feature/homebrew-setup`.
2. Push branch to fork (`rainbow`) and open PR against upstream.
3. Validate Homebrew install flow from tap in a clean environment.
4. Update release/tag strategy for formula URL pinning.

---
*Last updated: 2026-03-03*
