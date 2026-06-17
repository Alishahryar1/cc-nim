# Agent Governor

Runtime safety layer for agentic loops on weak-RLHF (open-weight) models.

Open-weight 80B–120B models (Qwen, Nemotron, DeepSeek, Kimi, GLM, GPT-OSS,
Mistral, Llama) have weaker termination judgment than RLHF-tuned proprietary
models (Claude, GPT-5). Without intervention they tool-call in tight loops and
never emit `stop_reason=end_turn`. The proxy then accumulates 400+ messages and
either burns through budgets or hangs on the upstream HTTP stream until the
provider finally times out the connection (we observed an 8.5-hour hang).

The governor inspects each `/v1/messages` request — after model routing, before
provider streaming — and chooses one of three actions:

| Action  | Meaning                                                     |
|---------|-------------------------------------------------------------|
| `pass`  | Forward the request unchanged.                               |
| `augment` | Forward, but with culled tools / appended system prompt.   |
| `abort` | Short-circuit with a forced "stop and answer" text response. |

Strong models (any id matching `GOVERNOR_STRONG_MODEL_PATTERNS`, default
`claude-`, `gpt-5`, `gpt-4`, `o1-`, `o3-`, `o4-`, `grok-`, `gemini-`) bypass the
governor entirely. The interventions are designed for weak models and would
hurt strong-model behaviour.

## Interventions

### Hard aborts (terminate the loop)

1. **Total tool-call budget** (`GOVERNOR_MAX_TOTAL_TOOL_CALLS`, default 40)
   Counts every `tool_use` block across the conversation. When the threshold
   is hit, returns an Anthropic-shape response asking the user to start a
   fresh session with a narrower task.
2. **Loop signature** (`GOVERNOR_LOOP_REPEAT_THRESHOLD`, default 3)
   The same tool name + the same arg hash, repeated N+ times in the recent
   tool-call window, returns an abort response telling the model to try a
   different approach.
3. **Consecutive tool-call cap** (`GOVERNOR_MAX_CONSECUTIVE_TOOL_CALLS`,
   default 8) — runs of `tool_use` (with intervening `tool_result` user
   messages) without an assistant text turn or fresh user prompt. When hit,
   tells the model to summarise what it has and stop.

### Augmentations (modify the request, then continue)

4. **Tool culling** (`GOVERNOR_TOOL_CULL_ENABLED`, default true,
   `GOVERNOR_TOOL_CULL_MAX` default 25)
   Free models drown in 91-tool inventories. Score each tool's name +
   description against tokens extracted from recent user messages, keep the
   top N plus a pinned "always keep" core (`read`, `write`, `edit`, `bash`,
   `grep`, `glob`, `ls`, `todowrite`, `taskcreate`, `exitplanmode`,
   `askuserquestion`, `webfetch`, `websearch`).
5. **Plan-mode preamble** (`GOVERNOR_PLAN_MODE_FOR_WEAK`, default true)
   Appends an instruction to the system prompt: *"Before calling any tool,
   write a numbered plan with at most 5 short steps."* The plan acts as an
   implicit termination signal — the model finishes the plan and stops.
6. **Termination hint** (`GOVERNOR_TERMINATION_HINT_FOR_WEAK`, default true)
   Tells the model that *text-only output is a valid answer*: "When you have
   answered or cannot proceed, output your final answer as plain text and
   stop. Do not keep calling tools to double-check." Counters the bias of
   weak models to keep tool-calling instead of writing a final answer.

## Size tiers

A 30B nano model and a 480B coder cannot share one cap schedule. The governor
classifies every non-frontier model into a **size tier** and uses tier-specific
defaults. Pattern matching against the model id picks the tier automatically;
manual overrides are available.

| Tier       | Approx params | Examples                                                    | max consecutive | max total | tool cull | plan-mode default | termination hint |
|------------|---------------|-------------------------------------------------------------|-----------------|-----------|-----------|-------------------|------------------|
| `tiny`     | ≤30B          | nemotron-3-nano-30b, gpt-oss-20b, llama-8b, gemma-9b        | 4               | 20        | 15        | yes               | yes              |
| `small`    | 30–80B        | qwen3-next-80b, llama-3.3-70b, nemotron-49b                 | 8               | 40        | 25        | yes               | yes              |
| `medium`   | 80–200B       | nemotron-super-120b, gpt-oss-120b, glm-4.6, kimi-k2         | 15              | 80        | 35        | yes               | yes              |
| `large`    | 200–400B      | llama-nemotron-ultra-253b, deepseek-v4-pro, qwen 235B       | 25              | 150       | 50        | no                | yes              |
| `xl`       | 400–600B      | qwen3-coder-480b, llama-3.1-405b, hermes-3-405b, ring-1t    | 80              | 400       | 80        | no                | yes              |
| `xxl`      | 600B+         | mistral-large-3-675b, future Qwen3-Max class                | 120             | 600       | 100       | no                | no               |
| `frontier` | n/a           | claude-, gpt-5, gemini, grok                                | bypass          | bypass    | bypass    | n/a               | n/a              |

Tier detection is substring-based, ordered largest → smallest. So
`mistral-large-3-675b-instruct-2512` matches `xxl` (via `675b`) before any
smaller pattern can match. MoE active-param suffixes like `a35b` are
deliberately ignored — total params drive the tier.

`large` and bigger turn off `plan-mode` by default because models in those
tiers usually plan well on their own; the preamble can hurt their output
quality. `xxl` also turns off `termination_hint` (frontier-adjacent reasoners
like mistral-large-3 generally terminate themselves).

## Configuration

All knobs are environment variables. Defaults are aggressive-but-safe and
tier-aware: setting a global env value overrides every tier; leaving it
unset lets each model use its tier's defaults.

```bash
# Master switch
GOVERNOR_ENABLED=true                       # set false to disable everywhere

# Hard limits — UNSET means "use tier defaults". Setting any of these
# overrides the tier default for ALL non-frontier models.
GOVERNOR_MAX_CONSECUTIVE_TOOL_CALLS=        # tier default (4–120)
GOVERNOR_MAX_TOTAL_TOOL_CALLS=              # tier default (20–600)
GOVERNOR_LOOP_REPEAT_THRESHOLD=3            # not tier-dependent
GOVERNOR_TOOL_CULL_ENABLED=true
GOVERNOR_TOOL_CULL_MAX=                     # tier default (15–100)

# Weak-model preambles — leave unset to follow per-tier defaults
GOVERNOR_PLAN_MODE_FOR_WEAK=                # tier default
GOVERNOR_TERMINATION_HINT_FOR_WEAK=         # tier default

# Per-tier override JSON. Adjusts the tier defaults themselves.
GOVERNOR_TIER_CAPS_JSON='{"xl":{"max_consecutive_tool_calls":60},"tiny":{"tool_cull_max":12}}'

# Model classification (comma-separated lowercase substring patterns)
GOVERNOR_WEAK_MODEL_PATTERNS=qwen,nemotron,deepseek-chat,deepseek-coder,kimi,moonshot,gpt-oss,glm,mistral,llama,wafer,ring,minimax,gemma,hermes,command,yi-
GOVERNOR_STRONG_MODEL_PATTERNS=claude-,gpt-5,gpt-4,o1-,o3-,o4-,grok-,gemini-

# Per-model overrides as a JSON object. Highest precedence.
# A "tier" field can pin a model into a specific tier explicitly.
GOVERNOR_OVERRIDES_JSON='{"nvidia/nemotron-3-super-120b-a12b":{"max_consecutive_tool_calls":5,"plan_mode":true},"qwen3-coder-480b":{"tier":"xxl"}}'
```

Override keys are matched as substrings against the request's model id; the
longest matching key wins. This lets you write a generic rule like
`{"qwen3-coder":{...}}` that catches every Qwen3 Coder variant.

**Resolution order (highest → lowest precedence):**
1. Per-model override (`GOVERNOR_OVERRIDES_JSON`)
2. Process-level env value (e.g. `GOVERNOR_MAX_CONSECUTIVE_TOOL_CALLS`)
3. Per-tier override (`GOVERNOR_TIER_CAPS_JSON`)
4. Tier defaults (the table above)

## Telemetry

Each request emits one structured log line:

```
GOVERNOR: request_id=req_abc model=qwen/qwen3-next-80b action=pass     consecutive_tools=2 total_tools=11 tools_in=91
GOVERNOR: request_id=req_abc model=qwen/qwen3-next-80b action=augment  consecutive_tools=3 total_tools=14 tools_in=91 tools_after_cull=25 intervention=tool_cull(-66),plan_mode,termination_hint
GOVERNOR: request_id=req_abc model=qwen/qwen3-next-80b action=abort    consecutive_tools=8 total_tools=27 tools_in=25 intervention=consecutive_tools reason=cap_reached
GOVERNOR: request_id=req_abc model=qwen/qwen3-next-80b action=abort    consecutive_tools=3 total_tools=14 tools_in=25 intervention=loop_signature reason=repeat:read
GOVERNOR: request_id=req_abc model=qwen/qwen3-next-80b action=abort    consecutive_tools=15 total_tools=40 tools_in=25 reason=total_tool_budget_exceeded
```

Strong-model bypass and disabled-state are emitted at DEBUG level only.

## Architecture

```
ClaudeProxyService.create_message
  ├── ModelRouter.resolve  (provider routing)
  ├── … existing checks …
  ├── AgentGovernor.review                  ← THIS MODULE
  │     ├── ConfigForModel
  │     ├── compute_stats(messages)         (counts + recent tool sigs)
  │     ├── HARD ABORTS:
  │     │     1. total tool budget
  │     │     2. loop signature
  │     │     3. consecutive cap
  │     ├── AUGMENTATIONS:
  │     │     4. tool culling
  │     │     5. plan-mode preamble
  │     │     6. termination hint
  │     └── log_decision()
  ├── try_optimizations  (existing — title-gen, prefix detection, …)
  └── provider.stream_response  (Anthropic ↔ OpenAI translation)
```

The governor is **stateless**: every decision is made from the request's
own `messages` array. Claude Code re-sends the full conversation each turn,
so no per-session storage is needed.

## Why a governor instead of fine-tuning the model?

Fine-tuning open-weight models for "good Claude-Code-like agent behaviour" is
expensive, slow, and fragile across model versions. The governor is:

* **External** — works with any model, any provider, any version.
* **Stateless** — no per-session memory; safe under restarts.
* **Reversible** — toggle via env, no behaviour change for strong models.
* **Cheap** — pure Python over a parsed pydantic request, no extra LLM calls.

Combined with the `claude-fcc` wrapper, this lets you get reliable agentic
loops out of free models that would otherwise burn 8 hours on overnight
infinite tool-call cycles.

## Limitations

* The governor cannot fix **NIM upstream flakiness** (502, RemoteProtocolError,
  ReadTimeout). What it does fix: those failures now happen *fast* because
  the abort path returns immediately rather than re-streaming endlessly.
* The keyword-based tool-relevance scorer is **simple by design**. If a tool's
  description is empty or generic, it scores poorly. If a project's MCP tool
  authors give better descriptions, culling improves automatically.
* The governor does **not** rewrite tool descriptions, fix tool schemas, or
  intervene mid-stream — only at request boundaries.
* **`AskUserQuestion`** is in the `ALWAYS_KEEP` list, so weak models can ask
  for clarification — but the model still has to *choose* to use it. The
  governor cannot force "ask the user" behaviour.

## Future work

* Embedding-based tool relevance (replace keyword scoring) — Phase 3 in the
  original proposal, deferred until keyword scoring proves insufficient.
* Per-MCP-server tool budgets (e.g. "at most 5 mempalace tools per request").
* Adaptive caps based on observed model behaviour over the past N requests
  (bandit-style: tighten caps for models that loop more often).
* Optional `task` / `todowrite` synthesis: if the model isn't using these
  tools but is in a long agentic chain, synthesise a fake "did you make a
  todo?" turn that nudges it.
