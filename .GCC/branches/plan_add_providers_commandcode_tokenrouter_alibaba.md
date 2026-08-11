# Execution Plan: Add CommandCode, TokenRouter, and Alibaba Providers

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: All existing tests pass without regression. Zero linter/formatter/type errors (`./scripts/ci.sh`).
- **Pre-requisites**: `uv` installed, Python 3.14.0 environment.

## 🛠️ Step-by-Step Sequence

### Step 1: Update Catalog, Settings, Admin Manifest, Defaults, Factory & Version
- [x] **Action**:
  - Add base URLs and descriptors in `config/provider_catalog.py` for `command_code`, `token_router`, and `alibaba`.
  - Re-export default base URLs in `providers/defaults.py`.
  - Add API keys & proxy fields in `config/settings.py`.
  - Add field overrides in `api/admin_config/provider_manifest.py`.
  - Register factory constructors in `providers/runtime/factory.py`.
  - Document keys in `.env.example`.
  - Bump semver in `pyproject.toml` (2.6.1 -> 2.7.0) and run `uv lock`.
- [x] **Verify**: `uv run ruff check` and `uv run ty check`
- **Verification Proof**:
```text
All selected CI checks passed.
Ban type ignore suppressions: passed
ruff format: passed (394 files unchanged)
ruff check: passed (All checks passed!)
ty check: passed (All checks passed!)
```

### Step 2: Implement Provider Classes
- [x] **Action**:
  - Create `providers/command_code/__init__.py` & `client.py` (`CommandCodeProvider`)
  - Create `providers/token_router/__init__.py` & `client.py` (`TokenRouterProvider`)
  - Create `providers/alibaba/__init__.py` & `client.py` (`AlibabaProvider`)
- [x] **Verify**: `uv run ruff check`
- **Verification Proof**:
```text
ruff format: passed (394 files unchanged)
ruff check: passed (All checks passed!)
```

### Step 3: Implement Unit Tests & Update Contract Tests
- [x] **Action**:
  - Create `tests/providers/test_command_code.py`
  - Create `tests/providers/test_token_router.py`
  - Create `tests/providers/test_alibaba.py`
  - Update `tests/contracts/test_provider_catalog_order.py` and `tests/api/test_admin.py`
- [x] **Verify**: `uv run pytest tests/providers/` and `./scripts/ci.sh`
- **Verification Proof**:
```text
All selected CI checks passed.
Ban type ignore suppressions: passed
ruff format: passed (394 files unchanged)
ruff check: passed (All checks passed!)
ty check: passed (All checks passed!)
pytest: 1737 passed, 8 skipped in 103s
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Base URL formatting or trailing slash inconsistencies.
- **Mitigation**: Ensure base_url handling uses standard `base_url or DEFAULT_BASE` pattern in `OpenAIChatTransport`.
