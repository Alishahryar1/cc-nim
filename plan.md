# First-Frame Gated Streaming PR Plan

## Summary

Fix issue #1020 at the real boundary: FCC currently commits downstream
`HTTP 200` as soon as FastAPI receives a `StreamingResponse`, before the
provider-backed stream has proven it can emit a valid first SSE chunk. If the
provider fails before that first chunk, Claude Code sees `200` with an empty or
malformed stream instead of a non-200 response it can retry.

The PR should introduce an API egress first-frame gate and align provider
transports so pre-start failures raise typed HTTP-mappable errors instead of
being converted into synthetic successful SSE streams. Once the first chunk has
escaped, HTTP status can no longer change, so post-start failures still need a
protocol-correct terminal stream frame.

Because this changes production API/provider behavior on `main`, bump the
current patch version from `3.4.12` to `3.4.13` unless `main` advances first,
then refresh `uv.lock`.

## Customer-Facing Contract

- `fcc-server` should not return `HTTP 200` for `/v1/messages` or
  `/v1/responses` until the stream can produce its first protocol chunk.
- Claude Code should receive a non-200 Anthropic-shaped error when upstream
  provider setup/retry fails before any stream output is viable.
- Codex should receive a non-200 OpenAI-shaped error when the Responses stream
  fails before `response.created`.
- After streaming has started, clients should receive a parseable terminal
  protocol frame instead of a truncated connection where feasible.
- Provider retries, midstream recovery, tool salvage, thinking/reasoning, tool
  calls, local web tools, non-streaming `/v1/messages`, and messaging behavior
  should remain unchanged except for the pre-start HTTP status fix.

## Grill-Me Decisions

### Is this caused by NIM or FCC?

Recommended answer: FCC owns the bug. NIM or any upstream can trigger the
failure by returning 429/5xx/504 or closing early, but FCC currently commits
downstream `HTTP 200` before upstream viability is known. Reproducing current
`api.response_streams.anthropic_sse_streaming_response()` shows
`http.response.start 200` is sent immediately, before the first delayed body
chunk and even when the body raises before its first chunk.

### Is a terminal SSE error frame enough?

Recommended answer: no. A terminal frame fixes only post-start truncation. It
does not restore Claude Code's HTTP retry behavior because the response is
still `HTTP 200`. The PR needs first-frame gating for pre-start failures plus
terminal framing for post-start failures.

### Should API egress own provider retry?

Recommended answer: no. Provider transports keep upstream retries, recovery,
tool salvage, and provider-specific fallbacks. API egress owns only the HTTP
commit boundary: do not commit success until there is a first chunk; after
success is committed, close the protocol cleanly if possible.

### Should providers keep emitting synthetic pre-start SSE errors?

Recommended answer: no. A provider-side final error before downstream-visible
output should raise a mapped `ProviderError`. Synthetic Anthropic SSE error
tails are only appropriate when the stream has already started or provider
state has produced output that must be closed in protocol shape.

### Should `/v1/responses` use a fresh assembler for egress failures?

Recommended answer: no. Responses streams are stateful. A post-start
`response.failed` must be produced by the same `ResponsesStreamAssembler` that
emitted `response.created`, preserving `response.id`, active output flushes,
usage, and response metadata.

## Architecture Target

### API Egress

`api/response_streams.py` should own public HTTP streaming commit timing.

Add an async first-frame helper that:

1. pulls the first chunk from an `AsyncIterator[str]` before constructing the
   public `StreamingResponse`;
2. returns a protocol JSON error response if the iterator raises before the
   first chunk;
3. returns a `StreamingResponse` that replays the first chunk and streams the
   rest when the first chunk exists;
4. wraps the post-first-chunk tail with a terminal-frame guard for unexpected
   non-cancellation failures.

The helper should be protocol-agnostic. Protocol-specific call sites provide:

- streaming headers;
- pre-start JSON error envelope builder;
- post-start terminal frame behavior.

Do not import `core/openai_responses` internals directly into API egress. API
handlers may use their adapter/facade objects.

### Provider Execution

`api/provider_execution.py` should keep resolving providers, preflight,
request tracing, raw-payload logging, token counting, and `traced_async_stream`.
It should not construct `StreamingResponse` and should not own protocol error
serialization.

### Provider Error Mapping

Provider transports need a single helper for pre-start final failures, for
example under `providers/error_mapping.py`, that converts any final stream
exception into a `ProviderError`:

- preserve existing mapped provider statuses for authentication, bad request,
  rate limit, overload, and upstream 5xx cases;
- preserve existing user-facing error-message sanitization and request-id
  appending;
- wrap internal stream exceptions such as `TruncatedProviderStreamError` in an
  upstream-style `APIError` rather than letting raw runtime exceptions escape;
- keep verbose raw exception detail behind existing diagnostic flags.

Do not make API egress inspect OpenAI/httpx exception classes directly. API
egress should receive either a first chunk or a typed exception it can serialize
for the product protocol.

### OpenAI-Chat Transport

`providers/transports/openai_chat/stream.py` should distinguish:

- **uncommitted pre-start final error**: raise mapped `ProviderError` so the API
  first-frame gate returns non-200;
- **committed/buffered stream failure**: preserve existing recovery/error-tail
  behavior;
- **early retry/recovery success**: unchanged;
- **complete tool salvage**: unchanged.

The important classification is not merely whether `message_start` was created
internally. It is whether anything has escaped the recovery holdback to the API
iterator. If the holdback has not committed and no buffered events are being
flushed as client-visible output, pre-start final errors should raise.

### Native Anthropic Transport

`providers/transports/anthropic_messages/stream.py` should apply the same
boundary:

- no committed/buffered downstream-visible event: raise mapped `ProviderError`;
- committed or buffered native stream: use native ledger error-tail behavior.

This keeps local native providers and future native providers consistent with
OpenAI-chat behavior.

### OpenAI Responses

`core/openai_responses/stream.py` and
`core/openai_responses/streaming/assembler.py` should own post-start Responses
terminal failures.

Do not add a stateless `OpenAIResponsesAdapter.egress_error_frame()` that mints
a fresh response id. Instead, the iterator returned by
`OpenAIResponsesAdapter.iter_sse_from_anthropic()` should:

- let pre-`response.created` failures propagate to API egress;
- after `response.created`, catch unexpected non-cancellation failures, call
  `ResponsesStreamAssembler.fail_response(...)` on the active assembler, yield
  the resulting `response.failed`, then re-raise or trace according to the
  existing egress tracing policy.

This preserves `response.failed.response.id == response.created.response.id`.

### Anthropic Messages

For `/v1/messages`, post-start unexpected failures may use a stateless terminal
Anthropic `event: error` as the final API egress fallback. Provider-owned error
tails remain preferred when provider code can close content blocks and emit
`message_delta`/`message_stop`.

## Implementation Plan

1. Add first-frame response helpers in `api/response_streams.py`.
   - Add a small private result type for either first chunk or pre-start
     exception.
   - Add `async def anthropic_sse_streaming_response(...)` or a new clearly
     named async builder, because first-frame probing must await the iterator.
   - Add the equivalent Responses builder with injected OpenAI error-envelope
     handling.
   - Keep wrappers protocol-agnostic and make call sites explicit.

2. Update `MessagesHandler._to_public_response()`.
   - Await the new Anthropic streaming response builder for `stream != false`.
   - Convert pre-start `ProviderError` to Anthropic JSON with the provider
     status code.
   - Convert unexpected pre-start exceptions to a safe 500 Anthropic JSON error
     using existing safe logging rules.
   - Keep `stream: false` aggregation unchanged.

3. Update `ResponsesHandler.create()`.
   - Await the new Responses streaming response builder.
   - Convert pre-start `ProviderError` to OpenAI-shaped JSON using
     `OpenAIResponsesAdapter.error_payload()`.
   - Convert unexpected pre-start exceptions to safe 500 OpenAI-shaped JSON.
   - Keep request conversion errors and `stream: false` rejection unchanged.

4. Add a provider-side pre-start failure exception path.
   - Introduce a small neutral helper in provider/shared error mapping that
     always returns a `ProviderError` for a final pre-start stream exception.
   - In OpenAI-chat final-error handling, if the recovery holdback is not
     committed and no event should be exposed, raise `map_error(...)` instead
     of yielding `emit_error_tail(...)`.
   - In native Anthropic final-error handling, raise mapped provider errors when
     no native event has been committed/buffered to the API.
   - Preserve existing synthetic SSE tails once events have escaped or must be
     closed.

5. Harden post-start terminal fallback.
   - Add Anthropic terminal-frame serialization under
     `core/anthropic/streaming/` only as a last-resort egress fallback.
   - In Responses stream conversion, add same-assembler failure handling for
     post-`response.created` exceptions.
   - Do not mint fresh Responses IDs for a terminal failure.

6. Update architecture docs.
   - Document that API egress owns first-frame HTTP commit gating.
   - Document that providers raise pre-start final failures but own retries and
     midstream recovery.
   - Document that Responses terminal failures are assembler-owned because
     Responses streams are stateful.

7. Bump version and lockfile.
   - Update `[project].version` from `3.4.12` to `3.4.13` unless `main`
     advances.
   - Run `uv lock`.

## Test Plan

### API Egress Tests

- ASGI-level test proving `http.response.start 200` is not sent until the first
  chunk is available.
- ASGI-level test proving a pre-first-chunk `ProviderError` returns non-200 JSON
  and sends no SSE body.
- ASGI-level test proving a pre-first-chunk unexpected exception returns safe
  500 JSON and does not leak raw exception text by default.
- ASGI-level test proving post-first-chunk exceptions yield a terminal frame
  before the exception closes the ASGI body.
- Cancellation and `GeneratorExit` tests proving no terminal frame is emitted
  into a dead socket.

### Messages API Tests

- `/v1/messages` provider pre-start `RateLimitError` returns HTTP 429
  Anthropic-shaped JSON.
- `/v1/messages` provider pre-start `APIError(status_code=504)` returns HTTP
  non-200 using the existing provider mapping, not HTTP 200.
- `/v1/messages` delayed valid provider stream still returns `text/event-stream`
  and valid Anthropic SSE.
- `/v1/messages stream:false` aggregation remains unchanged.

### Responses API Tests

- `/v1/responses` provider pre-start `RateLimitError` returns HTTP 429
  OpenAI-shaped JSON.
- `/v1/responses` delayed valid provider stream still returns
  `text/event-stream`.
- Responses post-start exception emits `response.failed` with the same
  `response.id` as `response.created`.
- Existing provider-emitted Anthropic `event:error` still converts to
  `response.failed` with the same active response id.

### Provider Transport Tests

- OpenAI-chat exhausted pre-stream 429/5xx/transport failure raises mapped
  `ProviderError` before any downstream-visible event.
- OpenAI-chat retry success path remains unchanged.
- OpenAI-chat midstream text failure still uses recovery/terminal tail behavior.
- OpenAI-chat complete tool salvage remains unchanged.
- Native Anthropic pre-send/pre-event failure raises mapped `ProviderError`.
- Native Anthropic midstream native event failure still closes through native
  recovery/error-tail behavior.

### Contract Tests

- API response stream helper is the only owner of first-frame HTTP commit
  gating.
- Responses egress terminal failure is assembler-owned, not generated by a
  stateless adapter helper.
- `api/provider_execution.py` remains free of `StreamingResponse` ownership.
- Architecture relative links still resolve.

## Verification Commands

Run targeted tests first:

```powershell
uv run pytest tests/api/test_response_streams.py tests/api/test_api_handlers.py tests/api/test_openai_responses.py
uv run pytest tests/providers/test_streaming_errors.py tests/providers/test_anthropic_messages.py tests/providers/test_openai_compat_5xx_retry.py tests/providers/test_anthropic_messages_429_retry.py
uv run pytest tests/core/openai_responses/test_sse.py tests/contracts/test_import_boundaries.py tests/contracts/test_architecture_contracts.py
```

Then run the final local gate:

```powershell
.\scripts\ci.ps1
```

## Risks And Guardrails

- Pulling the first chunk delays HTTP headers until upstream viability is known.
  This is intentional for Claude/Codex retry correctness, but tests should prove
  normal streams still begin promptly after the first provider chunk.
- A provider can emit a synthetic SSE error as its first chunk. The PR should
  avoid relying only on API first-chunk gating; providers must raise pre-start
  final failures instead of creating synthetic success streams.
- Responses post-start fallback must preserve assembler identity. Any test that
  shape-asserts only `response.failed` without comparing IDs is insufficient.
- Do not broaden API egress into retry/recovery ownership. That would erode the
  provider transport boundaries established in `ARCHITECTURE.md`.

## Out Of Scope

- Changing retry counts, backoff, or adding `Retry-After` support.
- Redesigning OpenAI-chat tool-call buffering.
- Changing model routing, thinking/reasoning policy, or provider request bodies.
- Changing messaging queue/cancellation behavior.
- Restoring non-streaming `/v1/responses`.
