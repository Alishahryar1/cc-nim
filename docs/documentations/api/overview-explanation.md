# Proxy Architecture & Event Loop Explanation

## The Problem: Interoperability Across Diverse AI Providers

Coding agents like Claude Code CLI interact via standard HTTP protocols, expecting the Anthropic Messages API format (`/v1/messages`) with Server-Sent Events (SSE) streaming. However, underlying AI providers—such as NVIDIA NIM, Ollama, OpenRouter, vLLM, or OpenAI—expose different API schemas, authentication methods, and streaming structures.

Free Claude Code bridges this gap by operating a high-performance local FastAPI proxy server that translates protocols dynamically while isolating client tooling from provider complexities.

## Architecture and Asynchronous Processing

The proxy server is built on FastAPI (`create_app` in `app.py`) and leverages Python's asynchronous event loop (`asyncio`) for non-blocking I/O operations.

```
┌──────────────────┐    Anthropic SSE    ┌──────────────────────────┐
│   Claude Code    │ ──────────────────► │  FastAPI Proxy Server    │
│      Client      │ ◄────────────────── │   (app.py / routes.py)   │
└──────────────────┘                     └────────────┬─────────────┘
                                                      │
                                           Acquire RequestRuntimeLease
                                                      ▼
┌──────────────────┐    Native / HTTP    ┌──────────────────────────┐
│ Provider Adapter │ ◄─────────────────► │     MessagesHandler      │
│ (NIM, Ollama...) │                     │ (Translation & Execution) │
└──────────────────┘                     └──────────────────────────┘
```

### 1. Request Lifecycle and Runtime Leases (`RequestRuntimeLease`)
When an incoming HTTP request hits an endpoint (such as `/v1/messages` or `/v1/responses`), the router acquires a request lease via `services.requests.acquire()`. This lease snapshots the active settings for the duration of the request, preventing configuration drift during ongoing streaming sessions. The lease is bound to the response lifecycle (`bind_response_lifetime`) and released automatically when streaming completes or encounters a network drop.

### 2. Protocol Translation and Event Normalization
Protocol handlers (`MessagesHandler` and `ResponsesHandler`) map request payloads into internal representations and translate provider outputs back into client-expected wire events:
- **Input Mapping**: Translates messages, tool definitions, system prompts, and reasoning parameters into the target provider's schema.
- **Output Normalization**: Converts provider streams into standardized Anthropic SSE events (`message_start`, `content_block_delta`, `message_stop`) or OpenAI Responses format.

## Architectural Tradeoffs

| Architecture | Strengths | Weaknesses |
| :--- | :--- | :--- |
| **Local Proxy Server (Selected)** | Full provider isolation, zero modifications to agent binaries, centralized logging and local admin UI. | Minor JSON serialization overhead per request. |
| **Direct SDK Patching** | No local network listener needed. | Fragile to upstream agent updates, cannot run independent web admin or multi-client routing. |

## Perspective and Design Principles

The proxy architecture enforces a strict one-way dependency flow: `UI / Presentation Layer (api/) -> Application Domain (application/) -> Infrastructure Adapters (providers/)`. By decoupling presentation from provider mechanics, the proxy guarantees defensive error handling, request trace correlation (`x-request-id`), and memory-safe execution across concurrent CLI sessions.
