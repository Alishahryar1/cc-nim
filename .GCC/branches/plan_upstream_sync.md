# Execution Plan: Reconcile Upstream Updates on upstream-sync Branch

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: All custom features of the `omni01-Cell/free-claude-code` fork (Google Antigravity CLI, Connected Account, Codex Desktop launcher `fcc-codex-desktop`, installers, context window settings) and all new upstream providers (Together AI, QwenCloud, xAI Grok, Novita AI, NaraRoute) are preserved and functioning without regression. 100% CI compliance (`./scripts/ci.sh`) must be maintained.
- **Pre-requisites**: `upstream/main` fetched (5 new commits). Working tree clean.

## 🛠️ Step-by-Step Sequence

### Step 1: Switch to `upstream-sync` branch and sync with `main`
- [ ] **Action**: `git checkout upstream-sync && git merge main`
- [ ] **Verify**: `git status`
- **Verification Proof**:
```text
[Pending]
```

### Step 2: Merge `upstream/main` into `upstream-sync`
- [ ] **Action**: `git merge upstream/main`
- [ ] **Verify**: Check for conflicts; resolve if any exist cleanly.
- **Verification Proof**:
```text
[Pending]
```

### Step 3: Run full CI test suite and linters
- [ ] **Action**: `./scripts/ci.sh`
- [ ] **Verify**: 100% pass across all 5 check jobs (suppression grep, ruff-format, ruff-check, ty, pytest).
- **Verification Proof**:
```text
[Pending]
```

### Step 4: Merge `upstream-sync` into `main`, bump semver if needed, update `uv.lock`, and push to `origin`
- [ ] **Action**: `git checkout main && git merge upstream-sync && git push origin main && git push origin upstream-sync`
- [ ] **Verify**: `git status && git log -n 5 --oneline`
- **Verification Proof**:
```text
[Pending]
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Merge conflicts in provider catalogs, docs, or pyproject.toml / uv.lock.
- **Mitigation**: Resolve conflicts manually, keeping both local and upstream provider definitions intact. Verify type checking with `uv run ty check` and tests with `uv run pytest`.
