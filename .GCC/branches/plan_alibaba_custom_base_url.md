# Execution Plan: Custom Base URL for Alibaba Provider

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Alibaba provider allows custom `ALIBABA_BASE_URL` via environment variable or Admin UI, falling back to `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. All tests pass (`./scripts/ci.sh`).

## 🛠️ Step-by-Step Sequence

### Step 1: Update Catalog & Settings
- [x] **Action**:
  - In `config/provider_catalog.py`, set `base_url_attr="alibaba_base_url"` on `alibaba` ProviderDescriptor.
  - In `config/settings.py`, add `alibaba_base_url: str = Field(default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", validation_alias="ALIBABA_BASE_URL")`.
  - In `.env.example`, document `ALIBABA_BASE_URL`.
  - Bump semver in `pyproject.toml` (2.7.0 -> 2.7.1) and run `uv lock`.
- [x] **Verify**: `uv run ruff check` and `uv run ty check`
- **Verification Proof**:
```text
All selected CI checks passed.
Ban type ignore suppressions: passed
ruff format: passed (394 files unchanged)
ruff check: passed (All checks passed!)
ty check: passed (All checks passed!)
```

### Step 2: Update Tests & Verify Admin Manifest
- [x] **Action**:
  - In `tests/providers/test_alibaba.py`, add tests for custom `base_url` configuration.
  - In `tests/api/test_admin.py`, verify `ALIBABA_BASE_URL` appears in fields.
- [x] **Verify**: `uv run pytest tests/providers/test_alibaba.py tests/api/test_admin.py` and `./scripts/ci.sh`
- **Verification Proof**:
```text
All selected CI checks passed.
Ban type ignore suppressions: passed
ruff format: passed (394 files unchanged)
ruff check: passed (All checks passed!)
ty check: passed (All checks passed!)
pytest: 1738 passed, 8 skipped in 103s
```
