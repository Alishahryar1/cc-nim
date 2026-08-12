<!--
Classification Reasoning:
1. Le lecteur cherche une vue d'ensemble conceptuelle et un index d'orientation pour le domaine Core & Runtime.
2. Le lecteur découvre l'organisation globale du domaine et ses composants principaux.
Type Diátaxis : Explanation (Index / Overview)
-->

# Core, Messaging & Runtime Domain

The Core, Messaging & Runtime domain provides the foundational concurrency, message state management, and provider lifecycle infrastructure for Free Claude Code.

## Domain Architecture

The domain is structured into three primary submodules:

- **`free_claude_code.core`**: Cross-cutting utilities including process locking (`InterprocessLock`), rate limiting (`TokenBucketRateLimiter`), diagnostic tracing (`trace_event`), failure taxonomy (`ProviderFailure`), and Anthropic protocol primitives.
- **`free_claude_code.messaging`**: Async message queue management (`TreeQueueManager`), message trees (`MessageTreeGraph`), session transcripts, voice intake (`Transcriber`), and command dispatching.
- **`free_claude_code.runtime`**: Server lifecycle management (`ApplicationRuntime`), provider generation leases (`ProviderRuntimeManager`), ASGI bootstrap, and configuration updates.

## Document Directory

This domain documentation follows the Diátaxis framework:

### Explanation

- [Messaging Queue & Tree Management](messaging-queue-explanation.md)
  Theoretical explanation of non-linear message graph trees (`TreeQueueManager`), request admission, turn idempotency, and asynchronous queue dequeuing.

### Reference

- [Provider Runtime & Generation Leases](runtime-leases-reference.md)
  Factual reference for `ProviderRuntimeManager`, immutable settings generations, lease acquisition/release semantics, and fault-isolated provider cleanup.

## Component Relationship

```
[ApplicationRuntime]
   ├── owns ──> [ProviderRuntimeManager] ── manages ──> [ProviderRuntime (Leases)]
   └── owns ──> [MessagingRuntime] ────── manages ──> [TreeQueueManager]
```

Requests acquire a generation lease from `ProviderRuntimeManager` to isolate provider configuration while `TreeQueueManager` maintains conversational graph integrity across async turns.
