# Product E2E Smoke Tests

`smoke/` is local-only. It can launch subprocesses, call real providers, touch
local model servers, and optionally send/delete bot messages. Hermetic contracts
belong under `tests/` and must stay green with plain `uv run pytest`.

## Taxonomy

- `smoke/prereq/`: liveness checks that prove the server, routes, auth, CLI
  scripts, provider pings, local `/models`, and bot permissions are reachable.
  These are prerequisites only.
- `smoke/product/`: end-to-end product scenarios. Feature smoke coverage comes
  from these tests, not from route/header/provider pings.
- `smoke/features.py`: source-of-truth feature map:
  feature -> subfeature -> scenario -> env -> expected behavior -> failure class.

## Required Local Commands

```powershell
uv run pytest smoke --collect-only -q
uv run pytest smoke -n 0 -s --tb=short
```

The second command skips everything unless `FCC_LIVE_SMOKE=1` is set, but still
writes skip entries to `.smoke-results/`.

## Product Smoke Run

```powershell
$env:FCC_LIVE_SMOKE = "1"
uv run pytest smoke -n 0 -s --tb=short
```

Provider smoke scenarios can run providers in parallel while preserving
sequential execution within each provider:

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "providers"
uv run pytest smoke -n auto --dist=loadgroup -s --tb=short
```

Provider product E2E runs once per configured provider, independent of `MODEL`,
`MODEL_FABLE`, `MODEL_OPUS`, `MODEL_SONNET`, and `MODEL_HAIKU`. Defaults come from the provider
catalog/docs and can be overridden with `FCC_SMOKE_MODEL_<PROVIDER>`, for example
`FCC_SMOKE_MODEL_DEEPSEEK=deepseek-v4-pro` (or `deepseek-v4-flash`). If no provider smoke model is
configured, live product smoke fails as `missing_env` unless you explicitly set
`FCC_ALLOW_NO_PROVIDER_SMOKE=1`.

## Targets

Default targets do not send real bot messages or load voice backends:

| Target | Product scenarios | Required environment |
| --- | --- | --- |
| `api` | messages, count_tokens full payload, errors, `/stop`, optimizations | configured provider only for streaming messages |
| `auth` | canonical bearer auth, conflicting legacy headers, invalid/missing auth | none; test sets an isolated token |
| `cli` | `fcc-init`, server entrypoint, Claude CLI adaptive thinking, session cleanup | Claude CLI binary and provider only for real CLI |
| `clients` | VS Code and JetBrains protocol payloads | configured provider |
| `config` | env precedence, removed-env migration, proxy/timeouts | none |
| `extensibility` | provider runtime and platform factory construction | none |
| `messaging` | fake Discord/Telegram full flow, literal clear scopes, trees, persistence, voice cancel | none |
| `providers` | multi-turn text, adaptive thinking history, tools, disconnect, errors | configured providers, optional `FCC_SMOKE_MODEL_*` |
| `tools` | forced tool_use and tool_result continuation | tool-capable configured provider |
| `rate_limit` | disconnect cleanup and follow-up request | configured provider |
| `lmstudio` | local `/models` plus OpenAI-chat-backed Messages through proxy | running LM Studio server |
| `llamacpp` | local `/models` plus OpenAI-chat-backed Messages through proxy | running llama-server |
| `ollama` | local `/v1/models` plus OpenAI-chat-backed Messages through proxy | running Ollama server |

Heavy/side-effectful targets are opt-in:

| Target | Product scenarios | Required environment |
| --- | --- | --- |
| `nvidia_nim_cli` | Claude Code CLI feature matrix across NIM models | `NVIDIA_NIM_API_KEY`, Claude CLI |
| `openrouter_free_cli` | Claude Code CLI feature matrix across OpenRouter free models | `OPENROUTER_API_KEY`, Claude CLI |
| `telegram` | getMe, send, edit, delete, optional manual inbound | token and chat/user ID |
| `discord` | channel access, send, edit, delete, optional manual inbound | token and channel ID |
| `voice` | generated WAV through local Whisper or NVIDIA NIM transcription | `VOICE_NOTE_ENABLED=true`, `FCC_SMOKE_RUN_VOICE=1` |

## Examples

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_PROVIDER_MATRIX = "open_router,nvidia_nim,deepseek,lmstudio,llamacpp,ollama"
uv run pytest smoke/product -n 0 -s --tb=short
```

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "ollama"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
uv run pytest smoke/prereq smoke/product -n 0 -s --tb=short
```

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "telegram,discord,voice"
$env:FCC_SMOKE_RUN_VOICE = "1"
uv run pytest smoke/product -n 0 -s --tb=short
```

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "nvidia_nim_cli"
$env:FCC_SMOKE_NIM_MODELS = "z-ai/glm-5.2,moonshotai/kimi-k2.6,minimaxai/minimax-m2.7,nvidia/nemotron-3-super-120b-a12b,deepseek-ai/deepseek-v4-pro,deepseek-ai/deepseek-v4-flash"
uv run pytest smoke/product -n 0 -s --tb=short
```

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "openrouter_free_cli"
$env:FCC_SMOKE_OPENROUTER_FREE_MODELS = "nvidia/nemotron-3-super-120b-a12b:free,openai/gpt-oss-120b:free,poolside/laguna-m.1:free"
uv run pytest smoke/product -n 0 -s --tb=short
```

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "messaging,config,extensibility"
uv run pytest smoke/product -n 0 -s --tb=short
```

## Environment

- `FCC_ENV_FILE`: explicit dotenv path for startup/config scenarios.
- `FCC_LIVE_SMOKE=1`: enables live smoke execution.
- `FCC_ALLOW_NO_PROVIDER_SMOKE=1`: permits no-provider live smoke for harness work.
- `FCC_SMOKE_TARGETS`: comma-separated targets, or `all`.
- `FCC_SMOKE_PROVIDER_MATRIX`: comma-separated provider prefixes to require.
- `FCC_SMOKE_MODEL_NVIDIA_NIM`, `FCC_SMOKE_MODEL_OPEN_ROUTER`,
  `FCC_SMOKE_MODEL_MISTRAL`, `FCC_SMOKE_MODEL_MISTRAL_REASONING`,
  `FCC_SMOKE_MODEL_MISTRAL_CODESTRAL`,
  `FCC_SMOKE_MODEL_DEEPSEEK`, `FCC_SMOKE_MODEL_KIMI`,
  `FCC_SMOKE_MODEL_WAFER`, `FCC_SMOKE_MODEL_MINIMAX`,
  `FCC_SMOKE_MODEL_OPENCODE`, `FCC_SMOKE_MODEL_OPENCODE_GO`,
  `FCC_SMOKE_MODEL_ZAI`, `FCC_SMOKE_MODEL_FIREWORKS`, `FCC_SMOKE_MODEL_CLOUDFLARE`,
  `FCC_SMOKE_MODEL_GEMINI`, `FCC_SMOKE_MODEL_GROQ`, `FCC_SMOKE_MODEL_CEREBRAS`,
  `FCC_SMOKE_MODEL_LMSTUDIO`,
  `FCC_SMOKE_MODEL_LLAMACPP`, `FCC_SMOKE_MODEL_OLLAMA`: optional per-provider
  smoke model overrides. Values may include the provider prefix or just the model
  name for that provider.
- `FCC_SMOKE_MODEL_MISTRAL_REASONING`: optional override for the dedicated
  Mistral native reasoning smoke, default `mistral/mistral-medium-3-5`.
- `FCC_SMOKE_NIM_MODELS`: optional comma-separated NVIDIA NIM CLI matrix models
  that replace the default characterization set.
- `FCC_SMOKE_NIM_EXTRA_MODELS`: optional comma-separated NVIDIA NIM CLI matrix
  models appended to the default or replacement set.
- `FCC_SMOKE_OPENROUTER_FREE_MODELS`: optional comma-separated OpenRouter free
  CLI matrix models that replace the default characterization set.
- `FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS`: optional comma-separated OpenRouter
  free CLI matrix models appended to the default or replacement set.
- `FCC_SMOKE_TIMEOUT_S`: per-request/subprocess timeout, default `45`.
- `FCC_SMOKE_CLAUDE_BIN`: Claude CLI executable name, default `claude`.
- `FCC_SMOKE_TELEGRAM_CHAT_ID`: Telegram chat/user ID for send/edit/delete.
- `FCC_SMOKE_DISCORD_CHANNEL_ID`: Discord channel ID for send/edit/delete.
- `FCC_SMOKE_INTERACTIVE=1`: enables manual inbound Telegram/Discord checks.
- `FCC_SMOKE_RUN_VOICE=1`: allows voice transcription backends to load/run.

## Windows / nested `uv run`

Run smoke the same way you run tests (`uv run pytest smoke` from the repo). Child
processes use the **same Python interpreter** as the test runner, not nested
`uv run`, so Windows does not try to replace `free-claude-code.exe` while it is
locked.

## Failure Classes

Smoke artifacts are written to `.smoke-results/` and redact env values whose
names contain `KEY`, `TOKEN`, `SECRET`, `WEBHOOK`, or `AUTH`.

- `missing_env`: required credentials, binary, provider config, local provider
  server/model, or opt-in flag is absent.
- `upstream_unavailable`: a real provider or bot API is not reachable.
- `probe_timeout`: the smoke driver reached the target, but the CLI/probe did
  not complete within the smoke timeout.
- `product_failure`: the app accepted the scenario but returned the wrong shape,
  crashed, leaked state, or violated the product contract.
- `harness_bug`: the smoke test or driver made an invalid assumption.
- `target_disabled`: skipped because `FCC_SMOKE_TARGETS` intentionally selected
  a different target.

`product_failure` and `harness_bug` are failures. `missing_env`,
`upstream_unavailable`, and `probe_timeout` are skips except when the user
explicitly selected a provider in `FCC_SMOKE_PROVIDER_MATRIX`;
selected-but-missing providers fail.
