# Execution Plan: Stateful Antigravity Tool Call Assembly

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Streaming `functionCall` events in `AntigravityProvider` must accumulate partial/empty argument chunks and deduplicate identical payloads per tool call, preventing duplicate tool block emissions to Claude Code CLI.
- **Pre-requisites**: All existing tests in `tests/providers/test_antigravity_client.py` passing cleanly.

## 🛠️ Step-by-Step Sequence

### Step 1: Refactor `AntigravityProvider.stream_response` for stateful tool call assembly
- [x] **Action**: Update `src/free_claude_code/providers/antigravity/client.py` to maintain `active_tool_by_name` tracking tool index, ID, and arguments across chunks.
- [x] **Verify**: `uv run pytest tests/providers/test_antigravity_client.py -v`
- **Verification Proof**:
```text
18 passed in 7.78s
```

### Step 2: Add comprehensive unit tests for empty-args-to-populated-args streaming
- [x] **Action**: Add unit test `test_stream_response_empty_args_then_populated_args_accumulated` in `tests/providers/test_antigravity_client.py`.
- [x] **Verify**: `uv run pytest tests/providers/test_antigravity_client.py -v`
- **Verification Proof**:
```text
tests/providers/test_antigravity_client.py::test_stream_response_empty_args_then_populated_args_accumulated PASSED [ 33%]
18 passed in 7.78s
```

### Step 3: CI & Versioning Verification
- [x] **Action**: Bump semver in `pyproject.toml` (v4.18.5), run `uv lock`, and execute `./scripts/ci.sh`.
- [x] **Verify**: `./scripts/ci.sh`
- **Verification Proof**:
```text
2889 passed, 69 skipped in 92.70s
All selected CI checks passed.
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Gemini emitting empty `{}` in chunk 1 and full args in chunk 2.
- **Mitigation**: Stateful assembler updates existing block index and emits delta instead of allocating a duplicate tool block.
