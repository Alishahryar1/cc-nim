# Graph Report - core  (2026-06-23)

## Corpus Check
- Corpus is ~13,958 words - fits in a single context window. You may not need a graph.

## Summary
- 468 nodes · 820 edges · 19 communities (16 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Anthropic SSE Formatting|Anthropic SSE Formatting]]
- [[_COMMUNITY_OpenAI Stream Assembly|OpenAI Stream Assembly]]
- [[_COMMUNITY_Anthropic-OpenAI Message Conversion|Anthropic-OpenAI Message Conversion]]
- [[_COMMUNITY_OpenAI Inputs and Tool Processing|OpenAI Inputs and Tool Processing]]
- [[_COMMUNITY_Stream Recovery Buffers|Stream Recovery Buffers]]
- [[_COMMUNITY_SSE Tracking and Emitted States|SSE Tracking and Emitted States]]
- [[_COMMUNITY_Thinking Tag and Tool Parsers|Thinking Tag and Tool Parsers]]
- [[_COMMUNITY_Tool Input Repair Algorithms|Tool Input Repair Algorithms]]
- [[_COMMUNITY_Native Messages Request Handling|Native Messages Request Handling]]
- [[_COMMUNITY_Native SSE Block Policies|Native SSE Block Policies]]
- [[_COMMUNITY_OpenAI Responses Adapters|OpenAI Responses Adapters]]
- [[_COMMUNITY_Stream Contracts and Server Tools|Stream Contracts and Server Tools]]
- [[_COMMUNITY_API Request and Logging Tracing|API Request and Logging Tracing]]
- [[_COMMUNITY_Anthropic Error Formatting|Anthropic Error Formatting]]
- [[_COMMUNITY_Sliding-Window Rate Limiting|Sliding-Window Rate Limiting]]
- [[_COMMUNITY_Token Usage Estimation|Token Usage Estimation]]
- [[_COMMUNITY_Core Initialization|Core Initialization]]
- [[_COMMUNITY_Thinking Rationale Helper|Thinking Rationale Helper]]
- [[_COMMUNITY_Conversion Helper Utils|Conversion Helper Utils]]

## God Nodes (most connected - your core abstractions)
1. `ResponsesStreamAssembler` - 33 edges
2. `SSEBuilder` - 30 edges
3. `EmittedNativeSseTracker` - 26 edges
4. `ResponsesConversionError` - 15 edges
5. `_append_input_item()` - 14 edges
6. `format_response_sse_event()` - 14 edges
7. `convert_messages()` - 13 edges
8. `get_block_type()` - 12 edges
9. `RecoveryHoldbackBuffer` - 11 edges
10. `ContentBlockManager` - 11 edges

## Surprising Connections (you probably didn't know these)
- `EmittedBlockState` --uses--> `SSEEvent`  [INFERRED]
  anthropic/emitted_sse_tracker.py → anthropic/stream_contracts.py
- `EmittedNativeSseTracker` --uses--> `SSEEvent`  [INFERRED]
  anthropic/emitted_sse_tracker.py → anthropic/stream_contracts.py
- `EmittedBlockState` --uses--> `ToolSchema`  [INFERRED]
  anthropic/emitted_sse_tracker.py → anthropic/stream_recovery.py
- `EmittedNativeSseTracker` --uses--> `ToolSchema`  [INFERRED]
  anthropic/emitted_sse_tracker.py → anthropic/stream_recovery.py
- `EmittedBlockState` --uses--> `SSEBuilder`  [INFERRED]
  anthropic/emitted_sse_tracker.py → anthropic/sse.py

## Communities (19 total, 3 thin omitted)

### Community 0 - "Anthropic SSE Formatting"
Cohesion: 0.06
Nodes (22): iter_provider_stream_error_sse_events(), Canonical Anthropic-style SSE sequence for provider-side streaming errors., Yield message_start (if needed), a text block with the error, then message_delta, ContentBlockManager, map_stop_reason(), _normalize_task_run_in_background(), SSE event builder for Anthropic-format streaming responses., Record tool name fragments as they arrive from chunked OpenAI streams. (+14 more)

### Community 1 - "OpenAI Stream Assembly"
Cohesion: 0.09
Nodes (23): AnthropicSseEvent, format_response_sse_event(), OpenAI Responses SSE event formatting., Format one OpenAI Responses SSE event., new_call_id(), new_message_item_id(), new_reasoning_item_id(), new_response_id() (+15 more)

### Community 2 - "Anthropic-OpenAI Message Conversion"
Cohesion: 0.09
Nodes (42): extract_text_from_content(), get_block_attr(), get_block_type(), Content block helpers for Anthropic-compatible payloads., Return a content block type when present., Extract concatenated text from message content., Get an attribute from a Pydantic model, lightweight object, or dict., AnthropicToOpenAIConverter (+34 more)

### Community 3 - "OpenAI Inputs and Tool Processing"
Cohesion: 0.11
Nodes (42): Errors and error envelopes for OpenAI Responses compatibility., Raised when a Responses request cannot be converted deterministically., ResponsesConversionError, _append_input_item(), _append_message_item(), _append_pending_reasoning(), _content_as_text(), _convert_message_content() (+34 more)

### Community 4 - "Stream Recovery Buffers"
Cohesion: 0.07
Nodes (22): How assistant reasoning history is replayed to OpenAI-compatible providers., ReasoningReplayMode, is_retryable_stream_error(), Return whether a provider stream error can be retried/recovered., Briefly hold downstream SSE so early stream cutoffs can be retried invisibly., Buffer ``event`` until holdback expires or cap is reached., Commit and return all held events., Drop held events without committing them downstream. (+14 more)

### Community 5 - "SSE Tracking and Emitted States"
Cohesion: 0.09
Nodes (13): EmittedBlockState, EmittedNativeSseTracker, Track content-block state for native Anthropic SSE strings we emit to clients., Next unused content block index based on emitted starts., Yield ``content_block_stop`` events for blocks that were started but not stopped, Tracked downstream block payload emitted to the client., Close dangling blocks, emit a text error block at a fresh index, then message ta, Parse emitted SSE frames so mid-stream errors can close blocks and pick a fresh (+5 more)

### Community 6 - "Thinking Tag and Tool Parsers"
Cohesion: 0.08
Nodes (18): ContentChunk, ContentType, Streaming parser for provider-emitted thinking tags., Parse content inside think tags., Flush any remaining buffered content., A chunk of parsed content., Streaming parser for ``<think>...</think>`` tags and generic variations (thought, Feed content and yield parsed chunks. (+10 more)

### Community 7 - "Tool Input Repair Algorithms"
Cohesion: 0.09
Nodes (28): accept_tool_json_repair(), _copied_messages(), make_native_text_recovery_body(), make_native_tool_repair_body(), make_openai_text_recovery_body(), make_openai_tool_repair_body(), parse_complete_tool_input(), Always-on recovery helpers for truncated provider streams. (+20 more)

### Community 8 - "Native Messages Request Handling"
Cohesion: 0.13
Nodes (22): _apply_openrouter_reasoning_policy(), build_base_native_anthropic_request_body(), build_openrouter_native_request_body(), dump_raw_messages_request(), _dump_request_fields(), _normalize_system_prompt_for_openrouter(), OpenRouterExtraBodyError, Native Anthropic Messages request body construction (JSON-ready dicts).  Provide (+14 more)

### Community 9 - "Native SSE Block Policies"
Cohesion: 0.12
Nodes (22): _allocate_new_segment(), _delta_type_to_block_kind(), format_native_sse_event(), is_terminal_openrouter_done_event(), NativeSseBlockPolicyState, parse_native_sse_event(), Shared native Anthropic SSE thinking policy, block remapping, and overlap repair, Close every open block except `current_upstream` and track duplicate upstream st (+14 more)

### Community 10 - "OpenAI Responses Adapters"
Cohesion: 0.12
Nodes (12): OpenAIResponsesAdapter, Facade for OpenAI Responses protocol adaptation., Convert between OpenAI Responses and the proxy's Anthropic core path., iter_sse_events(), parse_sse_event(), Anthropic SSE parsing used by the Responses stream adapter., openai_error_payload(), Return an OpenAI-compatible error envelope. (+4 more)

### Community 11 - "Stream Contracts and Server Tools"
Cohesion: 0.16
Nodes (10): SSE content_block ``type`` values for Anthropic web server tools (local handlers, _append_event(), assert_anthropic_stream_contract(), event_index(), parse_sse_lines(), parse_sse_text(), Neutral SSE parsing and Anthropic stream shape assertions.  Used by default CI c, Check minimal Anthropic-style SSE invariants: start/stop, block nesting.      Do (+2 more)

### Community 12 - "API Request and Logging Tracing"
Cohesion: 0.17
Nodes (15): api_messages_request_snapshot(), extract_claude_session_id_from_headers(), provider_chat_body_snapshot(), provider_native_messages_body_snapshot(), Structured TRACE events for end-to-end request / CLI / provider logging.  Emitte, Emit TRACE rows when a text stream completes, fails, cancels, or periodically., Sanitized OpenAI-compat chat body subset for traces (conversation text verbatim), Sanitized Anthropic Messages API body subset for traces. (+7 more)

### Community 13 - "Anthropic Error Formatting"
Cohesion: 0.29
Nodes (7): append_request_id(), format_user_error_preview(), get_user_facing_error_message(), User-facing error formatting shared by API, providers, and integrations., Return a readable, non-empty error message for users.      Known transport and O, Truncate a user-facing error string for short chat replies., Append request_id suffix when available.

### Community 14 - "Sliding-Window Rate Limiting"
Cohesion: 0.29
Nodes (3): Shared strict sliding-window rate limiting primitives., Strict sliding window limiter.      Guarantees: at most ``rate_limit`` acquisiti, StrictSlidingWindowLimiter

### Community 15 - "Token Usage Estimation"
Cohesion: 0.29
Nodes (5): estimate_text_tokens(), Usage helpers for OpenAI Responses payloads., Return a best-effort token estimate for Responses usage details., _TokenEncoder, Protocol

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ResponsesConversionError` connect `OpenAI Inputs and Tool Processing` to `Native Messages Request Handling`, `OpenAI Responses Adapters`?**
  _High betweenness centrality (0.361) - this node is a cross-community bridge._
- **Why does `SSEBuilder` connect `Anthropic SSE Formatting` to `SSE Tracking and Emitted States`?**
  _High betweenness centrality (0.260) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `SSEBuilder` (e.g. with `EmittedBlockState` and `EmittedNativeSseTracker`) actually correct?**
  _`SSEBuilder` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `EmittedNativeSseTracker` (e.g. with `SSEBuilder` and `SSEEvent`) actually correct?**
  _`EmittedNativeSseTracker` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ResponsesConversionError` (e.g. with `OpenAIResponsesAdapter` and `ResponsesToolIdentity`) actually correct?**
  _`ResponsesConversionError` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Shared strict sliding-window rate limiting primitives.`, `Strict sliding window limiter.      Guarantees: at most ``rate_limit`` acquisiti`, `Structured TRACE events for end-to-end request / CLI / provider logging.  Emitte` to the rest of the system?**
  _143 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Anthropic SSE Formatting` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._