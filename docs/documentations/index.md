# Free Claude Code (FCC) Documentation

Welcome to the official documentation for **Free Claude Code (FCC)**. This project implements a local AI proxy server connecting coding agents (Claude Code, OpenAI Codex, Pi) to diverse AI providers (Google Antigravity CLI, OpenAI, Anthropic, OpenRouter, NVIDIA NIM, Mistral, AgentRouter, CommandCode, TokenRouter, Alibaba DashScope, Together AI, QwenCloud, xAI Grok, Novita AI, NaraRoute, and local LLMs).

---

## 🏗️ High-Level Codebase Architecture

The codebase is organized into four core domains under `src/free_claude_code/`:

- **API & Application (`src/free_claude_code/api/`, `application/`)**: FastAPI web server, endpoint routing (`/v1/messages`, `/v1/responses`, `/admin`), SSE stream translation, and web search/fetch server tools.
- **CLI & Launchers (`src/free_claude_code/cli/`, `config/settings.py`)**: Entrypoints (`fcc-server`, `fcc-claude`, `fcc-codex`, `fcc-pi`), tray application, and hybrid ephemeral desktop launcher (`fcc-codex-desktop`).
- **AI Providers & Catalog (`src/free_claude_code/providers/`, `config/provider_catalog.py`)**: Provider abstractions, Google Antigravity OAuth 2.0 PKCE, tool call sanitation, reasoning effort policy, and SSE adapters.
- **Core, Messaging & Runtime (`src/free_claude_code/core/`, `messaging/`, `runtime/`)**: SDK-free canonical error handling, messaging tree queues (`TreeQueueManager`), and provider runtime leases (`ProviderRuntime`).

---

## 📚 Documentation Organization (Diátaxis Framework)

This documentation suite is structured into four distinct domains:

### 1. API & Web Server Tools (`docs/api/`)
* **[API Domain Index](api/index.md)**: Overview of API endpoints and request lifecycles.
* **[Proxy Architecture & Event Loop](api/overview-explanation.md)** *(Explanation)*: Technical breakdown of FastAPI async request processing and protocol translation.
* **[Endpoints & Web Tools Reference](api/endpoints-reference.md)** *(Reference)*: Specification of HTTP endpoints, headers, and server tools (`web_search`, `web_fetch`).
* **[Configuring Custom API Handlers](api/custom-handler-howto.md)** *(How-to Guide)*: Step-by-step guide for creating custom route handlers.

### 2. CLI & Desktop Integration (`docs/cli/`)
* **[CLI Domain Index](cli/index.md)**: Overview of CLI entrypoints and launchers.
* **[Getting Started with Coding Agents](cli/getting-started-tutorial.md)** *(Tutorial)*: Step-by-step tutorial for starting `fcc-server`, `fcc-claude`, `fcc-codex`, and `fcc-pi`.
* **[Launching Codex Desktop](cli/codex-desktop-howto.md)** *(How-to Guide)*: Guide for using `fcc-codex-desktop` with automatic TOML configuration.
* **[CLI Commands & Settings Reference](cli/cli-commands-reference.md)** *(Reference)*: Technical reference of CLI flags, environment variables, and configuration keys.

### 3. AI Providers & Model Catalog (`docs/providers/`)
* **[Providers Domain Index](providers/index.md)**: Overview of provider implementations.
* **[Provider Catalog Reference](providers/provider-catalog-reference.md)** *(Reference)*: Catalog of all supported providers, descriptors, and authentication schemas.
* **[Google Antigravity Architecture](providers/antigravity-explanation.md)** *(Explanation)*: In-depth guide on OAuth PKCE, Language Server fingerprinting, and SSE stream tool deduplication.
* **[Adding a New AI Provider](providers/add-provider-howto.md)** *(How-to Guide)*: Step-by-step extension guide for implementing new provider adapters.
* **[Reasoning Effort Policy](providers/reasoning-policy-explanation.md)** *(Explanation)*: Architectural explanation of reasoning effort mapping across model families.

### 4. Core Protocol, Messaging & Runtime (`docs/core_runtime/`)
* **[Core & Runtime Domain Index](core_runtime/index.md)**: Overview of core queue and execution leases.
* **[Messaging Tree Queue Architecture](core_runtime/messaging-queue-explanation.md)** *(Explanation)*: Explanation of `TreeQueueManager`, message deduplication, and transcription event flows.
* **[Provider Runtime & Leases Reference](core_runtime/runtime-leases-reference.md)** *(Reference)*: Specifications for `ProviderRuntime`, resource lease management, and canonical error domains.
