# Execution Plan: Reconcile Upstream Updates on upstream-sync Branch

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: All custom features of the `omni01-Cell/free-claude-code` fork (Google Antigravity CLI, Connected Account, Codex Desktop launcher `fcc-codex-desktop`, installers, context window settings) and all new upstream providers (Together AI, QwenCloud, xAI Grok, Novita AI, NaraRoute) are preserved and functioning without regression. 100% CI compliance (`./scripts/ci.sh`) must be maintained.
- **Pre-requisites**: `upstream/main` fetched (5 new commits). Working tree clean.

## 🛠️ Step-by-Step Sequence

### Step 1: Switch to `upstream-sync` branch and sync with `main`
- [x] **Action**: `git checkout upstream-sync && git merge main`
- [x] **Verify**: `git status`
- **Verification Proof**:
```text
Basculement sur la branche 'upstream-sync'
Mise à jour cef7bc2..628e906
Fast-forward
 .GCC/branches/plan_upstream_sync.md | 43 +++++++++++++++++++++++++++++++++++++
 .GCC/main.md                        | 10 ++++-----
 .GCC/resume.md                      | 28 ++++++++++--------------
 3 files changed, 59 insertions(+), 22 deletions(-)
```

### Step 2: Merge `upstream/main` into `upstream-sync`
- [x] **Action**: `git merge upstream/main`
- [x] **Verify**: Check for conflicts; resolve if any exist cleanly.
- **Verification Proof**:
```text
[upstream-sync d13327b] Merge remote-tracking branch 'upstream/main' into upstream-sync
Resolved conflicts in: pyproject.toml, src/free_claude_code/config/provider_catalog.py, src/free_claude_code/config/settings.py, tests/contracts/test_provider_catalog_order.py, tests/providers/test_provider_runtime.py, uv.lock.
```

### Step 3: Run full CI test suite and linters
- [x] **Action**: `./scripts/ci.sh`
- [x] **Verify**: 100% pass across all 5 check jobs (suppression grep, ruff-format, ruff-check, ty, pytest).
- **Verification Proof**:
```text
==> Ban suppressions and legacy annotations
==> ruff format: 546 files left unchanged
==> ruff check --fix: All checks passed!
==> ty check: All checks passed!
==> pytest: 2968 passed, 69 skipped in 89.60s (0:01:29)
All selected CI checks passed.
```

### Step 4: Merge `upstream-sync` into `main`, bump semver if needed, update `uv.lock`, and push to `origin`
- [x] **Action**: `git checkout main && git merge upstream-sync && git push origin main && git push origin upstream-sync`
- [x] **Verify**: `git status && git log -n 5 --oneline`
- **Verification Proof**:
```text
Merged upstream-sync into main successfully and pushed to origin/main and origin/upstream-sync.
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Merge conflicts in provider catalogs, docs, or pyproject.toml / uv.lock.
- **Mitigation**: Resolve conflicts manually, keeping both local and upstream provider definitions intact. Verify type checking with `uv run ty check` and tests with `uv run pytest`.
