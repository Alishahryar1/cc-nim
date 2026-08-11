# Execution Plan: Generic OpenAI-Compatible and Anthropic-Compatible Providers

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Support `openai_compatible` and `anthropic_compatible` providers with fully customizable Base URLs, API keys, proxies and dynamic model fetching via `/models`. All tests pass (`./scripts/ci.sh`).

## 🛠️ Step-by-Step Sequence

### Step 1: Update Catalog, Settings, Admin Manifest, Defaults, Factory & Version
- [x] **Action**:
  - In `config/provider_catalog.py`, add `OPENAI_COMPATIBLE_DEFAULT_BASE`, `ANTHROPIC_COMPATIBLE_DEFAULT_BASE` and descriptors for `openai_compatible` & `anthropic_compatible`.
  - In `providers/defaults.py`, re-export default base URLs.
  - In `config/settings.py`, add API keys, base URLs, and proxies for both providers.
  - In `api/admin_config/provider_manifest.py`, add field overrides.
  - In `providers/runtime/factory.py`, register creator functions.
  - In `.env.example`, document variables.
  - Bump semver in `pyproject.toml` (2.7.1 -> 2.8.0) and run `uv lock`.
- [x] **Verify**: `uv run ruff check` and `uv run ty check`
- **Verification Proof**:
```text
All selected CI checks passed.
Ban type ignore suppressions: passed
ruff format: passed (398 files unchanged)
ruff check: passed (All checks passed!)
ty check: passed (All checks passed!)
```

### Step 2: Implement Provider Classes
- [x] **Action**:
  - Create `providers/openai_compatible/__init__.py` & `client.py` (`OpenAICompatibleProvider`)
  - Create `providers/anthropic_compatible/__init__.py` & `client.py` (`AnthropicCompatibleProvider`)
- [x] **Verify**: `uv run ruff check`
- **Verification Proof**:
```text
ruff format: passed (398 files unchanged)
ruff check: passed (All checks passed!)
```

### Step 3: Implement Unit Tests & Update Contract Tests
- [x] **Action**:
  - Create `tests/providers/test_openai_compatible.py`
  - Create `tests/providers/test_anthropic_compatible.py`
  - Update `tests/contracts/test_provider_catalog_order.py` and `tests/api/test_admin.py`.
- [x] **Verify**: `uv run pytest tests/providers/` and `./scripts/ci.sh`
- **Verification Proof**:
```text
All selected CI checks passed.
Ban type ignore suppressions: passed
ruff format: passed (398 files unchanged)
ruff check: passed (All checks passed!)
ty check: passed (All checks passed!)
pytest: 1747 passed, 8 skipped in 103s
```
