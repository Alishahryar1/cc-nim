# Task 1.1: Add `enable_native_tool_approvals` Setting

## Task Summary
Add a new `enable_native_tool_approvals: bool = True` setting to the FCC Settings class. This feature flag controls whether native Claude Desktop tool approval dialogs are used instead of FCC's auto-intercepts.

## Location in Plan
Phase 1: Native Tool Approvals (Issue 1) — First task, foundational for Tasks 1.2-1.4

## Files to Modify
- **Modify:** `src/free_claude_code/config/settings.py`
- **Test:** `tests/config/test_settings.py`

## Interfaces
- **Produces:** `Settings.enable_native_tool_approvals: bool` (default `True`)
- **Consumed by:** Tasks 1.2 and 1.3 (intercept methods in messages.py)

## Exact Test Code

```python
# tests/config/test_settings.py
def test_settings_enable_native_tool_approvals_default_true():
    from free_claude_code.config.settings import Settings
    s = Settings()
    assert s.enable_native_tool_approvals is True

def test_settings_enable_native_tool_approvals_configurable():
    from free_claude_code.config.settings import Settings
    s = Settings(enable_native_tool_approvals=False)
    assert s.enable_native_tool_approvals is False
```

## Exact Implementation Code

```python
# src/free_claude_code/config/settings.py
# Add to Settings class:
enable_native_tool_approvals: bool = True
```

## Commands to Run

```bash
# Step 2: Verify test fails
pytest tests/config/test_settings.py::test_settings_enable_native_tool_approvals_default_true -v

# Step 4: Verify test passes after implementation
pytest tests/config/test_settings.py::test_settings_enable_native_tool_approvals_default_true -v

# Step 5: Commit
git add src/free_claude_code/config/settings.py tests/config/test_settings.py
git commit -m "feat: add enable_native_tool_approvals setting"
```

## Global Constraints (from plan)
- All features opt-in via Settings (fcc.toml) — feature flags default to safe values
- No breaking changes to existing API — backward compatible
- Follow existing code style: no `from __future__`, use `contextlib.suppress`, type hints
- CI must pass: suppressions, ruff-format, ruff-check, ty, pytest
- Trace events for all new operations with `request_id` correlation

## Notes
- Setting defaults to `True` (enabled) per design spec — native tool approvals ON by default
- This is a simple pydantic field addition to existing Settings class
- No migration needed; existing configs get new default