# API & Application Domain Documentation

Welcome to the **API & Application** domain documentation for Free Claude Code. This domain encompasses the FastAPI proxy server, HTTP endpoint routing, protocol translation between AI provider formats, administration UI API, and outbound web server tools (`web_search`, `web_fetch`).

## Navigation & Map

The documentation for this domain is organized according to the Diátaxis framework across three specialized guides:

1. **[Proxy Architecture & Event Loop Explanation](overview-explanation.md)** *(Explanation)*
   Learn about the FastAPI proxy design, asynchronous event loop, execution lease lifecycle (`RequestRuntimeLease`), and bidirectional LLM protocol translation.

2. **[Endpoints & Web Server Tools Reference](endpoints-reference.md)** *(Reference)*
   Consult factual technical specifications for all HTTP endpoints (`/v1/messages`, `/v1/responses`, `/v1/models`, `/admin`), HTTP headers, request schemas, and web tool execution constraints.

3. **[How to Create and Register a Custom API Handler](custom-handler-howto.md)** *(How-to Guide)*
   Follow step-by-step instructions to implement a custom route handler under `api/handlers/` and register its route in the FastAPI server.

## Codebase Map

- **`src/free_claude_code/api/app.py`**: Pure FastAPI application factory (`create_app`), middleware stack, and global exception handlers.
- **`src/free_claude_code/api/routes.py`**: Primary router defining public API endpoints (`/v1/messages`, `/v1/responses`, `/v1/models`, `/health`, `/stop`).
- **`src/free_claude_code/api/admin_routes.py`**: Local admin UI router serving loopback-restricted configuration and status APIs (`/admin`, `/admin/api/*`).
- **`src/free_claude_code/api/handlers/`**: Protocol handler implementations (`MessagesHandler`, `ResponsesHandler`, `TokenCountHandler`).
- **`src/free_claude_code/api/web_tools/`**: Outbound web tools (`web_search`, `web_fetch`) with DNS pinning and egress safety policies.
- **`src/free_claude_code/application/`**: Core application logic including runtime lease management (`execution.py`) and provider routing (`routing.py`).
