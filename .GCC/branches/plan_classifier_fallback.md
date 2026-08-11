# Execution Plan: Safety Classifier Fallback Routing

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Any safety classifier request (detected by `is_safety_classifier_request`) must be routed to a configured/default lightweight model (e.g. `gemini/gemini-3.1-flash-lite` or `openai/gpt-5.4-nano`) with `thinking_enabled=False`, without affecting standard message requests.
- **Pre-requisites**: A clean codebase state with all 99 tests currently passing.

## 🛠️ Step-by-Step Sequence

### Step 1: Configuration Additions
- **Action**: Add `MODEL_CLASSIFIER` configuration to `config/settings.py` and `.env.example`.
- **Verify**: Validate that the settings load correctly by running pytest on settings.
- **Verification Proof**:
```text
```

### Step 2: Routing Logic Implementation
- **Action**: Modify `_apply_message_routing_policies` in `api/handlers/messages.py` to route safety classifier requests to the configured `MODEL_CLASSIFIER` (or provider-specific default like `gemini-3.1-flash-lite` for Gemini or `gpt-5.4-nano` for OpenAI/OpenAI-compatible) and force `thinking_enabled=False`.
- **Verify**: Run python checks and unit tests.
- **Verification Proof**:
```text
```

### Step 3: Test Updates & Final Code Verification
- **Action**: Update `tests/api/test_api_handlers.py` to mock and assert the new routing behavior for safety classifier requests. Run `ruff format`, `ruff check --fix` and `pytest`.
- **Verify**: `pytest -v tests/api/test_api_handlers.py` and `./scripts/ci.sh` (or `pytest`) showing all checks and tests pass with 0 warnings/errors.
- **Verification Proof**:
```text
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: The configured `MODEL_CLASSIFIER` points to a provider for which the user doesn't have an API key or configuration.
- **Mitigation**: If the resolved provider lacks credentials or fails, fallback to the session's primary model (with thinking disabled) or raise a clean error.
