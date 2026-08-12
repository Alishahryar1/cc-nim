<!--
Classification Reasoning:
1. Le lecteur cherche des informations factuelles et techniques précises (signatures, classes, baux) sans guide pas-à-pas.
2. Le lecteur applique des connaissances existantes et a besoin de consulter la référence de l'API ProviderRuntime.
Type Diátaxis : Reference
-->

# ProviderRuntime & Generation Leases — Reference

`ProviderRuntimeManager` and `ProviderRuntime` provide single-owner generation tracking, lazy provider instantiation, and safe lifecycle leasing for LLM provider clients.

## Classes & Syntax

```python
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.provider_manager import (
    ProviderGenerationLease,
    ProviderRuntimeManager,
)
```

## ProviderRuntimeManager Methods

| Method | Return Type | Description |
|---|---|---|
| `acquire()` | `Coroutine[ProviderGenerationLease]` | Acquires an active lease on the current provider generation. Raises `ApplicationUnavailableError` if shutting down. |
| `current_settings()` | `Settings` | Returns the immutable `Settings` snapshot bound to the active generation. |
| `cached_model_ids()` | `dict[str, frozenset[str]]` | Returns synchronized cached model IDs per provider. |
| `replace_settings(new_settings)` | `Coroutine[None]` | Retires the active generation and publishes a new generation with updated settings. |
| `close()` | `Coroutine[None]` | Drains active leases and cleans up all provider instances across generations. |

## ProviderGenerationLease Interface

| Attribute / Method | Type | Description |
|---|---|---|
| `generation_id` | `int` (property) | Monotonically increasing generation identifier. |
| `settings` | `Settings` (property) | Settings snapshot for this lease duration. |
| `resolve_provider(provider_id)` | `BaseProvider` | Returns cached provider or lazily constructs it via factory. |
| `release()` | `Coroutine[None]` | Decrements active lease count and triggers cleanup if generation is retired and drained. |

## Request Lifecycle & Isolation Behavior

1. **Lease Acquisition**: A request acquires a `ProviderGenerationLease` via `async with manager.acquire()`.
2. **Provider Resolution**: `lease.resolve_provider("nvidia_nim")` creates or retrieves the provider instance bound to the lease's settings generation.
3. **Hot-Reload Isolation**: When settings update, `replace_settings()` marks old generation as `retired=True`. Existing requests hold their leases until completion; new requests acquire the new generation.
4. **Cleanup & Fault Isolation**: Once `active_leases` drops to zero on a retired generation, `runtime.cleanup()` closes provider instances asynchronously. Exceptions during cleanup are aggregated into `ExceptionGroup`.

## Example

```python
async with provider_manager.acquire() as lease:
    provider = lease.resolve_provider("anthropic")
    response = await provider.complete(request)
```

## Notes

- Generation leases are context managers (`__aenter__` / `__aexit__`) guaranteeing release even on exceptions.
- Shutting down the manager waits for active leases to drain before closing network sockets.
