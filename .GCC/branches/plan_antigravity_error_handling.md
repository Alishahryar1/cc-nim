# Execution Plan: Fix Antigravity Streaming Error Handling & Translation

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Any upstream error (HTTP 429, 401, 500 or network exception) occurring during Google Antigravity direct REST streaming MUST yield a valid Anthropic SSE error stream without escaping into ASGI/Starlette, avoiding `RuntimeError: Caught handled exception, but response already started.`. Error JSON payloads must be parsed into clear, human-readable strings.
- **Pre-requisites**: Existing tests in `tests/providers/` passing.

## 🛠️ Step-by-Step Sequence

### Step 1: Implement JSON Error Message Extraction and Stream Exception Handling in `providers/antigravity/client.py`
- [x] **Action**: Add `_extract_error_message()` helper to parse Google RPC error JSON, update `_raise_mapped_http_error()`, and wrap `stream_response()` in `try...except` to yield Anthropic SSE error events using `iter_provider_stream_error_sse_events`.
- [x] **Verify**: `uv run ruff check providers/antigravity/client.py` && `uv run ty check`
- **Verification Proof**:
```text
uv run ruff check providers/antigravity/client.py -> All checks passed!
uv run ty check -> All checks passed!
```

### Step 2: Add Comprehensive Unit Tests for Antigravity Streaming Errors
- [x] **Action**: Update/add tests in `tests/providers/test_antigravity_client.py` to cover HTTP 429 (Quota Exhausted JSON), HTTP 401, and Google RPC JSON error parsing.
- [x] **Verify**: `uv run pytest tests/providers/test_antigravity_client.py -v`
- **Verification Proof**:
```text
============================== 12 passed in 4.55s ==============================
```

### Step 3: Run Full Local CI Suite
- [x] **Action**: Run `./scripts/ci.sh` to ensure all 5 CI checks (Ruff format, Ruff check, Ty, Pytest, Ban type ignore suppressions) pass cleanly.
- [x] **Verify**: `./scripts/ci.sh`
- **Verification Proof**:
```text
Ban type ignore suppressions: passed
ruff format: passed
ruff check: passed
ty check: passed
pytest: 1792 passed, 8 skipped in 51.29s
All selected CI checks passed.
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Over-wrapping `stream_response` could obscure legitimate bugs if exceptions aren't logged.
- **Mitigation**: Log exceptions with `logger.error("Antigravity streaming error: %s", error_message)` before yielding SSE error events.
