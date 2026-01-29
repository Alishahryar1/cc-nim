# cc-nim

Use **Claude Code CLI for free** with NVIDIA NIM's free unlimited 40 reqs/min API. This lightweight proxy converts Claude Code's Anthropic API requests to NVIDIA NIM format.

## Quick Start

### 1. Get Your Free NVIDIA API Key

1. Visit [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
2. Sign in or create a free NVIDIA account
3. Generate a new API key (`nvapi-...`)

### 2. Install & Configure

```bash
git clone https://github.com/Alishahryar1/cc-nim.git
cd cc-nim

cp .env.example .env
```

Edit `.env`:

```dotenv
NVIDIA_NIM_API_KEY=nvapi-your-key-here
BIG_MODEL=moonshotai/kimi-k2-instruct
SMALL_MODEL=moonshotai/kimi-k2-instruct
```

### 3. Run

**Terminal 1 - Start the proxy:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2 - Run Claude Code:**

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses NVIDIA NIM for free.

## Available Models

See [`nvidia_nim_models.json`](nvidia_nim_models.json) for the full list of supported models.

Popular choices:

- `moonshotai/kimi-k2.5`
- `z-ai/glm4.7`
- `minimaxai/minimax-m2.1`
- `mistralai/devstral-2-123b-instruct-2512`

Browse all models at [build.nvidia.com](https://build.nvidia.com/explore/discover)

### Updating the Model List

To update `nvidia_nim_models.json` with the latest models from NVIDIA NIM, run the following command:

```bash
curl "https://integrate.api.nvidia.com/v1/models" > nvidia_nim_models.json
```

## Configuration

| Variable                 | Description                    | Default                               |
| ------------------------ | ------------------------------ | ------------------------------------- |
| `NVIDIA_NIM_API_KEY`     | Your NVIDIA API key            | required                              |
| `BIG_MODEL`              | Model for Sonnet/Opus requests | `moonshotai/kimi-k2-thinking`         |
| `SMALL_MODEL`            | Model for Haiku requests       | `moonshotai/kimi-k2-thinking`         |
| `NVIDIA_NIM_BASE_URL`    | NIM endpoint                   | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_NIM_RATE_LIMIT`  | Requests per window            | `40`                                  |
| `NVIDIA_NIM_RATE_WINDOW` | Rate limit window (seconds)    | `60`                                  |

See [`.env.example`](.env.example) for all supported parameters.

## Development

### Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

### Adding Your Own Provider

Extend `BaseProvider` in `providers/` to add support for other APIs:

```python
from providers.base import BaseProvider, ProviderConfig

class MyProvider(BaseProvider):
    async def complete(self, request):
        # Make API call, return raw JSON
        pass

    async def stream_response(self, request, input_tokens=0):
        # Yield Anthropic SSE format events
        pass

    def convert_response(self, response_json, original_request):
        # Convert to Anthropic response format
        pass
```
