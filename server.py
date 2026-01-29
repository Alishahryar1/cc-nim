"""
Claude Code Proxy - NVIDIA NIM Implementation

This server acts as a robust proxy between Anthropic API requests and NVIDIA NIM,
enabling Claude Code CLI to utilize NIM models with full support for:
- Streaming with SSE (Server-Sent Events)
- Thinking/Reasoning blocks and Reasoning-Split mode
- Native and heuristic tool use parsing
- Automatic model mapping (Haiku/Sonnet/Opus to NIM equivalents)
- Fast prefix detection for CLI policy specifications
"""

import time
import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, field_validator, model_validator
from providers.nvidia_nim import NvidiaNimProvider, ProviderConfig
from providers.exceptions import ProviderError
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import tiktoken
from providers.claude_cli import CLISession, CLIParser

# Optional: telethon for the bot
try:
    from telethon import TelegramClient, events
except ImportError:
    TelegramClient = None
    events = None

# Initialize tokenizer
ENCODER = tiktoken.get_encoding("cl100k_base")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log", encoding="utf-8", mode="w")],
)
logger = logging.getLogger(__name__)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# =============================================================================
# Models
# =============================================================================

BIG_MODEL = os.getenv("BIG_MODEL", "moonshotai/kimi-k2-instruct")
SMALL_MODEL = os.getenv("SMALL_MODEL", "moonshotai/kimi-k2-instruct")


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
                ContentBlockThinking,
            ]
        ],
    ]
    reasoning_content: Optional[str] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    extra_body: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @model_validator(mode="after")
    def map_model(self) -> "MessagesRequest":
        if self.original_model is None:
            self.original_model = self.model

        clean_v = self.model
        for prefix in ["anthropic/", "openai/", "gemini/"]:
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix) :]
                break

        if "haiku" in clean_v.lower():
            self.model = SMALL_MODEL
        elif "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
            self.model = BIG_MODEL

        if self.model != self.original_model:
            logger.debug(f"MODEL MAPPING: '{self.original_model}' -> '{self.model}'")

        return self


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None

    @field_validator("model")
    @classmethod
    def validate_model_field(cls, v, info):
        clean_v = v
        for prefix in ["anthropic/", "openai/", "gemini/"]:
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix) :]
                break

        if "haiku" in clean_v.lower():
            return SMALL_MODEL
        elif "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
            return BIG_MODEL
        return v


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[
        Union[
            ContentBlockText, ContentBlockToolUse, ContentBlockThinking, Dict[str, Any]
        ]
    ]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# =============================================================================
# Provider
# =============================================================================

provider_config = ProviderConfig(
    api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
    base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    rate_limit=int(os.getenv("NVIDIA_NIM_RATE_LIMIT", "40")),
    rate_window=int(os.getenv("NVIDIA_NIM_RATE_WINDOW", "60")),
)

# Global provider instance for DI
_provider: Optional[NvidiaNimProvider] = None


def get_provider() -> NvidiaNimProvider:
    global _provider
    if _provider is None:
        _provider = NvidiaNimProvider(provider_config)
    return _provider


# =============================================================================
# FastAPI App
# =============================================================================

tele_client: Optional["TelegramClient"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tele_client
    try:
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        if TelegramClient and api_id and api_hash:
            logger.info("Starting Telegram Bot...")
            session_path = os.path.join(WORKSPACE_PATH, "claude_bot.session")
            tele_client = TelegramClient(session_path, int(api_id), api_hash)

            # Register handlers BEFORE starting
            register_bot_handlers(tele_client)

            await tele_client.start()
            asyncio.create_task(tele_client.run_until_disconnected())

            # Notify user
            try:
                await tele_client.send_message(
                    "me", f"🚀 **Claude unified server is online!** (v{app.version})"
                )
            except:
                pass
            logger.info("Bot started and online message sent.")
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        tele_client = None

    yield
    if tele_client:
        await tele_client.disconnect()
    logger.info("Server shutting down...")
    global _provider
    if _provider and hasattr(_provider, "_client"):
        await _provider._client.aclose()


# =============================================================================
# Telegram Bot & CLI Configuration
# =============================================================================

WORKSPACE_PATH = os.path.abspath(os.getenv("CLAUDE_WORKSPACE", "agent_workspace"))
ALLOWED_DIRS = []
raw_dirs = os.getenv("ALLOWED_DIRS", "")
if raw_dirs:
    # Handle Windows backslash corrosion (\a, \b etc) by replacing them
    for d in raw_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        # If it looks like a Windows path with corrupted escapes, try to fix
        fixed = (
            d.replace("\a", "\\a")
            .replace("\b", "\\b")
            .replace("\f", "\\f")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace("\v", "\\v")
        )
        ALLOWED_DIRS.append(os.path.normpath(fixed))
# Internal URL for the CLI to use (points to this server)
INTERNAL_API_URL = "http://localhost:8082/v1"

# Initialize Global Instances
cli_session = CLISession(WORKSPACE_PATH, INTERNAL_API_URL, ALLOWED_DIRS)


def register_bot_handlers(client: "TelegramClient"):
    ALLOWED_USER_ID = os.getenv("ALLOWED_TELEGRAM_USER_ID")
    logger.info(f"DEBUG: Registering bot handlers. Allowed user ID: {ALLOWED_USER_ID}")

    @client.on(events.NewMessage())
    async def handle_telegram_message(event):
        sender_id = str(event.sender_id)
        logger.info(f"BOT_EVENT: From {sender_id} | Text: {event.text[:50]}")

        target_id = str(ALLOWED_USER_ID).strip()
        if sender_id != target_id:
            logger.debug(f"BOT_SECURITY: Ignored message from {sender_id}")
            return
        
        # 1. Handle Commands
        if event.text == "/stop":
            await cli_session.stop()
            await event.reply("⏹ **Claude process stopped.**")
            return
            
        # 2. Filter out bot's own status messages and empty text
        if not event.text or any(event.text.startswith(p) for p in ["⏳", "�", "🔧", "✅", "❌", "🚀", "🤖"]):
            return

        logger.info(f"BOT_TASK: {event.text}")
        status_msg = await event.reply("⏳ **Launching Claude CLI...**")

        current_content = ""
        last_ui_update = 0

        async def update_bot_ui(text, status=None):
            nonlocal last_ui_update
            if not text and not status:
                return
            now = time.time()
            if now - last_ui_update < 0.8: # Slightly faster
                return
            try:
                display = f"{status}\n\n{text}" if status else text
                if len(display) > 4000:
                    display = "..." + display[-3997:]
                await status_msg.edit(display, parse_mode="markdown")
                last_ui_update = now
            except:
                pass

        try:
            async for event_data in cli_session.start_task(event.text):
                if not isinstance(event_data, dict):
                    continue

                parsed = CLIParser.parse_event(event_data)

                if event_data.get("type") == "raw":
                    raw_line = event_data.get("content")
                    if not raw_line: continue
                    if "login" in raw_line.lower():
                        await event.reply("⚠️ **Claude requires login. Run `claude` in terminal.**")
                    continue

                if not parsed:
                    continue

                if parsed["type"] == "thinking":
                    # Display thinking tokens with special prefix
                    thinking_text = parsed["text"]
                    await event.reply(f"💭 **Thinking:**\n```\n{thinking_text[:1500]}{'...' if len(thinking_text) > 1500 else ''}\n```", parse_mode="markdown")

                elif parsed["type"] == "content":
                    current_content += parsed["text"]
                    await update_bot_ui(current_content, "🧠 **Claude is working...**")
                
                elif parsed["type"] == "tool_start":
                    names = [t.get("name") for t in parsed["tools"]]
                    # Send a SEPARATE message for tool calls as requested
                    await event.reply(f"🔧 **Running Tools:** `{', '.join(names)}`")
                    # Update status message to show we're working
                    await update_bot_ui(current_content, "⏳ **Executing tools...**")
                
                elif parsed["type"] == "subagent_start":
                    tasks = parsed["tasks"]
                    await event.reply(f"🤖 **Subagent Task:** {', '.join(tasks)}")
                    await update_bot_ui(current_content, "🔎 **Subagent working...**")

                elif parsed["type"] == "complete":
                    if parsed.get("status") == "failed":
                        await status_msg.edit(f"❌ **Failed**\n\n{current_content}", parse_mode="markdown")
                    else:
                        await status_msg.edit(f"✅ **Complete**\n\n{current_content or 'Done.'}", parse_mode="markdown")
                
                elif parsed["type"] == "error":
                    await event.reply(f"❌ **Error:** {parsed['message']}")
        except Exception as e:
            logger.error(f"Bot failed: {e}")
            await event.reply(f"💥 **Failed:** {e}")


FAST_PREFIX_DETECTION = os.getenv("FAST_PREFIX_DETECTION", "true").lower() == "true"


app = FastAPI(title="Claude Code Proxy", version="2.0.0", lifespan=lifespan)


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    """Handle provider-specific errors and return Anthropic format."""
    logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_anthropic_format(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle general errors and return Anthropic format."""
    logger.error(f"General Error: {str(exc)}")
    import traceback

    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "An unexpected error occurred.",
            },
        },
    )


def extract_command_prefix(command: str) -> str:
    import shlex

    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command)
        if not parts:
            return "none"

        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if "=" in part and not part.startswith("-"):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {
            "git",
            "npm",
            "docker",
            "kubectl",
            "cargo",
            "go",
            "pip",
            "yarn",
        }

        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            if not second_word.startswith("-"):
                return f"{first_word} {second_word}"
            return first_word
        return first_word if not env_prefix else " ".join(env_prefix) + " " + first_word

    except ValueError:
        return command.split()[0] if command.split() else "none"


def is_prefix_detection_request(request_data: MessagesRequest) -> tuple[bool, str]:
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, ""

    msg = request_data.messages[0]
    content = ""
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            if hasattr(block, "text"):
                content += block.text

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except Exception:
            pass

    return False, ""


def get_token_count(messages, system=None, tools=None) -> int:
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += len(ENCODER.encode(block.text))

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                # Handle dictionary or Pydantic model
                b_type = getattr(block, "type", None)

                if b_type == "text":
                    total_tokens += len(ENCODER.encode(getattr(block, "text", "")))
                elif b_type == "thinking":
                    # Thinking tokens are part of context if they are in history
                    total_tokens += len(ENCODER.encode(getattr(block, "thinking", "")))
                elif b_type == "tool_use":
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {})
                    # Add tokens for definitions
                    total_tokens += len(ENCODER.encode(name))
                    total_tokens += len(ENCODER.encode(json.dumps(inp)))
                    total_tokens += 10  # Control tokens approximate
                elif b_type == "tool_result":
                    content = getattr(block, "content", "")
                    if isinstance(content, str):
                        total_tokens += len(ENCODER.encode(content))
                    else:
                        total_tokens += len(ENCODER.encode(json.dumps(content)))
                    total_tokens += 5  # Control tokens approximate

    if tools:
        for tool in tools:
            # Approximate tool definition tokens
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(ENCODER.encode(tool_str))

    # Add some overhead for message formatting (approx 3 tokens per message)
    total_tokens += len(messages) * 3
    if tools:
        total_tokens += len(tools) * 5  # Extra overhead for tool definitions

    return max(1, total_tokens)


def log_request_details(request_data: MessagesRequest):
    """Log detailed request content for debugging."""

    def sanitize(text: str, max_len: int = 200) -> str:
        """Escape newlines and truncate for single-line logging."""
        text = text.replace("\n", "\\n").replace("\r", "\\r")
        return text[:max_len] + "..." if len(text) > max_len else text

    for i, msg in enumerate(request_data.messages):
        role = msg.role
        if isinstance(msg.content, str):
            logger.debug(f"  [{i}] {role}: {sanitize(msg.content)}")
        elif isinstance(msg.content, list):
            text_acc = []
            for block in msg.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_acc.append(getattr(block, "text", ""))
                else:
                    if text_acc:
                        logger.debug(
                            f"  [{i}] {role}/text: {sanitize(''.join(text_acc))}"
                        )
                        text_acc = []
                    if block_type == "tool_use":
                        name = getattr(block, "name", "unknown")
                        inp = getattr(block, "input", {})
                        logger.debug(
                            f"  [{i}] {role}/tool_use: {name}({sanitize(json.dumps(inp), 500)})"
                        )
                    elif block_type == "tool_result":
                        content = getattr(block, "content", "")
                        tool_use_id = getattr(block, "tool_use_id", "unknown")
                        logger.debug(
                            f"  [{i}] {role}/tool_result[{tool_use_id}]: {sanitize(str(content))}"
                        )
                    elif block_type == "thinking":
                        thinking = getattr(block, "thinking", "")
                        logger.debug(f"  [{i}] {role}/thinking: {sanitize(thinking)}")
            if text_acc:
                logger.debug(f"  [{i}] {role}/text: {sanitize(''.join(text_acc))}")


@app.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    provider: NvidiaNimProvider = Depends(get_provider),
):
    try:
        if FAST_PREFIX_DETECTION:
            is_prefix_req, command = is_prefix_detection_request(request_data)
            if is_prefix_req:
                import uuid

                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=request_data.model,
                    content=[{"type": "text", "text": extract_command_prefix(command)}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=5),
                )

        logger.info(
            f"Request: model={request_data.model}, messages={len(request_data.messages)}, stream={request_data.stream}"
        )
        log_request_details(request_data)

        if request_data.stream:
            input_tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            return StreamingResponse(
                provider.stream_response(request_data, input_tokens=input_tokens),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response_json = await provider.complete(request_data)
            return provider.convert_response(response_json, request_data)

    except ProviderError:
        # Re-raise ProviderError to be handled by the specialized exception handler
        raise
    except Exception as e:
        import traceback

        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    try:
        return TokenCountResponse(
            input_tokens=get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "status": "ok",
        "provider": "nvidia_nim",
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "bot_running": tele_client is not None}


@app.post("/stop")
async def stop_cli():
    stopped = await cli_session.stop()
    return {"status": "terminated" if stopped else "no_active_process"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="debug")
