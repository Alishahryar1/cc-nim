# Endpoints & Web Tools Reference

This document provides technical specifications for all HTTP endpoints exposed by the Free Claude Code FastAPI proxy server, along with specifications for the integrated outbound web server tools (`web_search` and `web_fetch`).

## Global HTTP Headers

| Header | Required | Description |
| :--- | :--- | :--- |
| `x-api-key` | Conditional | Authentication key validated when proxy authorization is enabled (`require_proxy_auth`). |
| `anthropic-version` | Required (`/v1/messages`) | Anthropic API protocol version string (e.g., `2023-06-01`). |
| `x-request-id` | Optional / Injected | Unique request correlation identifier assigned or forwarded by middleware. |
| `x-claude-code-session-id` | Optional | Client session identifier extracted for diagnostic trace context. |

## Core Public Endpoints

### 1. `POST /v1/messages`
Processes Anthropic Messages API creation requests.

- **Request Schema**: `MessagesRequest` (`model`, `messages`, `max_tokens`, `tools`, `stream`, `system`, `temperature`).
- **Response**:
  - `stream=false`: Standard Anthropic JSON `Message` object.
  - `stream=true`: Server-Sent Events stream (`text/event-stream`).

```json
// Example Request Payload
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024,
  "stream": false
}
```

### 2. `POST /v1/responses`
Processes OpenAI Responses API compatible requests.

- **Request Schema**: `OpenAIResponsesRequest` (`model`, `input`, `instructions`, `tools`, `stream`).
- **Response**: JSON or SSE stream formatted according to OpenAI Responses specifications.

### 3. `POST /v1/messages/count_tokens`
Calculates estimated token consumption without invoking model generation.

- **Request Schema**: `TokenCountRequest`.
- **Response**: `{"input_tokens": 42}`.

### 4. Service Utility Endpoints

| Endpoint | Method | Response | Description |
| :--- | :--- | :--- | :--- |
| `/v1/models` | `GET` | `ModelsListResponse` | Lists all configured and dynamically discovered model IDs. |
| `/health` | `GET` | `{"status": "healthy"}` | Service health check probe. |
| `/` | `GET` | `{"status": "ok", "provider": "...", "model": "..."}` | Returns current proxy status and active model. |
| `/stop` | `POST` | `{"status": "stopped", "cancelled_count": N}` | Halts active CLI sessions and pending queue tasks. |

## Administration Endpoints (`/admin`)

Access to `/admin` routes is strictly restricted to local loopback clients (`127.0.0.1`, `::1`, `localhost`).

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/admin` | `GET` | Serves the single-page local administration UI (`index.html`). |
| `/admin/api/config` | `GET` | Retrieves current configuration fields and values. |
| `/admin/api/config/validate` | `POST` | Validates proposed configuration changes against administrative manifests. |
| `/admin/api/config/apply` | `POST` | Persists configuration changes and triggers service restart if required. |
| `/admin/api/status` | `GET` | Returns runtime health, provider readiness, and system metrics. |
| `/admin/api/providers/{id}/test` | `POST` | Executes a connectivity and authentication test against a specific provider. |
| `/admin/api/models/refresh` | `POST` | Triggers active model discovery across connected providers. |

## Web Server Tools Reference

### `web_search`
Queries DuckDuckGo Lite to fetch search results.
- **Parameters**: `query` (string).
- **Output Format**: List of dictionaries containing `title`, `snippet`, and `url`.
- **Constraints**: Results capped at `_MAX_SEARCH_RESULTS` (15); response body limited to `_MAX_WEB_FETCH_RESPONSE_BYTES` (2 MB).

### `web_fetch`
Fetches and extracts plain text from a target web URL.
- **Parameters**: `url` (string).
- **Output Schema**: `{"url": str, "title": str, "media_type": "text/plain", "data": str}`.
- **Egress & Security Constraints**:
  - **IP Filtering**: Enforces `WebFetchEgressPolicy` blocking loopback, private RFC-1918, and link-local IP addresses.
  - **DNS Pinning**: Uses static IP resolution (`_PinnedEgressStaticResolver`) to prevent DNS rebinding attacks.
  - **Redirect Cap**: Maximum of 5 HTTP redirects allowed (`_MAX_WEB_FETCH_REDIRECTS`).
  - **Character Limit**: Extracted text truncated at `_MAX_FETCH_CHARS` (100,000 characters).
