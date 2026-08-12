# Execution Plan: Deduplicate Antigravity Stream Tool Calls

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Emitting SSE stream responses in `AntigravityProvider` must never yield duplicate `tool_use` blocks for identical `functionCall` instances within a single stream response.
- **Pre-requisites**: Existing tests in `tests/providers/test_antigravity_client.py` passing cleanly.

## 🛠️ Step-by-Step Sequence

### Step 1: Implement tool call deduplication in `AntigravityProvider.stream_response`
- [x] **Action**: Modify `src/free_claude_code/providers/antigravity/client.py` to maintain a set of seen tool call signatures `(fn_name, json.dumps(fn_args, sort_keys=True))` per stream response turn, skipping duplicate `functionCall` SSE events.
- [x] **Verify**: `uv run pytest tests/providers/test_antigravity_client.py -v`
- **Verification Proof**:
```text
16 passed in 10.75s
```

### Step 2: Add unit tests for duplicate tool call streaming in `test_antigravity_client.py`
- [x] **Action**: Add unit test `test_stream_response_duplicate_tool_calls_deduplicated` in `tests/providers/test_antigravity_client.py` verifying that repeated `functionCall` parts across multiple SSE chunks emit exactly one `tool_use` block.
- [x] **Verify**: `uv run pytest tests/providers/test_antigravity_client.py -k "test_stream_response_duplicate_tool_calls_deduplicated" -v`
- **Verification Proof**:
```text
tests/providers/test_antigravity_client.py::test_stream_response_duplicate_tool_calls_deduplicated PASSED [100%]
1 passed in 6.07s
```

### Step 3: CI & Versioning Verification
- [x] **Action**: Bump semver in `pyproject.toml` (v4.18.4), run `uv lock`, and execute `./scripts/ci.sh`.
- [x] **Verify**: `./scripts/ci.sh`
- **Verification Proof**:
```text
2888 passed, 69 skipped in 74.07s
All selected CI checks passed.
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Multiple distinct tool calls in the same turn (e.g. `Edit(file1)` and `Edit(file2)`).
- **Mitigation**: Deduplication key includes serialized arguments (`fn_args`), ensuring distinct tool calls have distinct signatures and are both emitted.
