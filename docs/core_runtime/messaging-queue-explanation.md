<!--
Classification Reasoning:
1. Le lecteur réfléchit aux concepts d'architecture (modèle théorique et garanties d'idempotence) plutôt qu'à exécuter des étapes pratiques.
2. Le lecteur découvre le fonctionnement interne de la gestion d'arbres de messages pour la première fois.
Type Diátaxis : Explanation
-->

# Continuous messaging trees with TreeQueueManager — How it works

## Context

In multi-channel conversational applications, managing message state with simple linear queues introduces severe race conditions and lost context when users reply to older turns or issue concurrent prompts. Traditional sequential execution locks the entire chat session, blocking incoming interactions and failing to represent non-linear conversation threads cleanly.

Free Claude Code introduces `TreeQueueManager` to solve thread tracking, claim processing, and state isolation across multiple platform scopes.

## How it works

`TreeQueueManager` maintains an in-memory directed acyclic graph (`MessageTreeGraph`) for each active conversation root. Incoming messages undergo strict admission control before entering execution.

```
Root Message (Node 0)
 ├── User Prompt A (Node 1) ── [Status ID 1] ──> Claimed & Executing
 └── User Prompt B (Node 2) ── [Status ID 2] ──> Enqueued (Pending)
```

1. **Admission & Idempotence**: When `admit()` receives an `IncomingMessage`, it inspects reference mappings (`node_id` and `status_message_id`). Duplicate submissions are intercepted immediately and routed to `enqueue_or_claim()`, returning an existing claim or queue position without duplicate side effects.
2. **Reply Resolution**: Replies reference parent nodes by logical ID or status message ID. `TreeQueueManager` attaches child nodes directly to their resolved parents, branching execution trees safely.
3. **Queue Processing & Dequeuing**: Claims are dispatched to `NodeProcessor`. When a node finishes, `complete_claim()` updates state snapshots and automatically dequeues the next eligible child node in the tree branch.

## Why it is this way

The tree aggregate pattern delivers key operational invariants:

- **Strict Turn Idempotence**: Assigning unique status message IDs per prompt ensures network retries never trigger duplicate LLM requests.
- **Cascading Cancellation**: Cancelling a parent node gracefully halts dependent child execution claims through `_drain_cancelled_tasks()`.
- **Fault Isolation**: Interrupted claims transition to failed states while keeping parent historical context valid for subsequent retries.

## Alternatives and tradeoffs

| Architecture | Strengths | Weaknesses |
|---|---|---|
| **Flat FIFO Queue** | Simple implementation | Rejects parallel turns; corrupts context on branch replies |
| **Global Session Lock** | Zero concurrency bugs | High latency; blocks unrelated channels |
| **Tree Queue (`TreeQueueManager`)** | Concurrent branching, strict idempotence, non-blocking admission | Higher internal complexity; requires explicit cleanup |

## Further reading

- [Runtime Leases Reference](runtime-leases-reference.md)
- [Core & Runtime Index](index.md)
