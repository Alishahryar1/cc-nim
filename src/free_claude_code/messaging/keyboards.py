"""Interactive inline keyboard builders for Telegram Bot."""

import typing
from typing import Any

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
except ImportError:
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None


import hashlib

_callback_registry = {}

def register_callback_path(prefix: str, path: str) -> str:
    """Register a path and return a unique short callback data key under 64 bytes."""
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    key = f"{prefix}:{h}"
    _callback_registry[key] = path
    return key

def get_registered_path(key: str) -> str | None:
    return _callback_registry.get(key)


def make_stop_keyboard(node_id: str) -> Any:
    """Build a keyboard with a Stop button for the given task node."""
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return None
    keyboard = [
        [
            InlineKeyboardButton("⏹ Stop Task", callback_data=f"stop_task:{node_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def make_clear_confirm_keyboard() -> Any:
    """Build a confirmation keyboard for clearing conversation history."""
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return None
    keyboard = [
        [
            InlineKeyboardButton("✅ Yes, Clear All", callback_data="clear_confirm"),
            InlineKeyboardButton("❌ Cancel", callback_data="clear_cancel"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def make_start_keyboard() -> Any:
    """Build the main menu inline keyboard for the /start greeting."""
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return None
    keyboard = [
        [
            InlineKeyboardButton("📁 Browse Workspace", callback_data="workspace_ls:"),
            InlineKeyboardButton("⚙️ Settings Panel", callback_data="menu_settings"),
        ],
        [
            InlineKeyboardButton("🤖 Select AI Model", callback_data="model_p:0:"),
            InlineKeyboardButton("🧹 Clear Chat History", callback_data="menu_clear"),
        ],
        [
            InlineKeyboardButton("⏹ Stop Active Tasks", callback_data="menu_stop"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


ALL_MODELS = [
    # Page 1: Recommended / Top Models
    ("Grok 4.5 🚀 (ZenMux)", "zenmux/x-ai/grok-4.5-free"),
    ("DeepSeek Flash ⚡ (Iamhc)", "iamhc/DeepSeek-V4-Flash"),
    ("MiniMax M2.7 🤖 (Iamhc)", "iamhc/MiniMax-M2.7"),
    ("GLM 4 9B 🧠 (Nvidia NIM)", "nvidia_nim/z-ai/glm-4-9b-chat"),
    ("Nemotron 3 🦾 (Nvidia NIM)", "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"),
    
    # Page 2: OpenAI-Compatible / Gateway
    ("Codestral Latest (Mistral)", "mistral_codestral/codestral-latest"),
    ("DeepSeek Chat (DeepSeek)", "deepseek/deepseek-chat"),
    ("OpenRouter Free (OpenRouter)", "open_router/openrouter/free"),
    ("Command R+ (Cohere)", "cohere/command-a-plus-05-2026"),
    ("GPT-4o (GitHub Models)", "github_models/openai/gpt-4.o"),

    # Page 3: Gateway & Gemini
    ("Gemini 2.5 Flash (Gemini)", "gemini/models/gemini-2.5-flash"),
    ("Llama 3.3 70B (Groq)", "groq/llama-3.3-70b-versatile"),
    ("DeepSeek V4 Pro (Wafer)", "wafer/DeepSeek-V4-Pro"),
    ("Kimi K2.5 (Kimi)", "kimi/kimi-k2.5"),
    ("MiniMax M3 (MiniMax)", "minimax/MiniMax-M3"),

    # Page 4: Inference Providers
    ("Qwen 72B (Sambanova)", "sambanova/Qwen2.5-72B-Instruct"),
    ("Llama 3.3 70B (Fireworks)", "fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct"),
    ("Kimi K2.6 (Cloudflare)", "cloudflare/@cf/moonshotai/kimi-k2.6"),
    ("GLM 5.2 (Z.ai)", "zai/glm-5.2"),
    ("Mistral Small (Mistral)", "mistral/mistral-small-latest"),

    # Page 5: Local Providers
    ("LM Studio Local", "lmstudio/<model-id>"),
    ("llama.cpp Local", "llamacpp/<model-id>"),
    ("Ollama Local", "ollama/<model-tag>"),
]


def make_model_keyboard(current_model: str, page: int = 0, search_query: str = "") -> tuple[str, Any]:
    """Build a paginated model selection keyboard, highlighting selection and supporting search."""
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return "", None

    # Filter models if there is a search query
    filtered = ALL_MODELS
    if search_query:
        query_lower = search_query.strip().lower()
        filtered = [
            (name, path) for name, path in ALL_MODELS
            if query_lower in name.lower() or query_lower in path.lower()
        ]

    PAGE_SIZE = 5
    total_items = len(filtered)
    total_pages = max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE)

    # Clamp page index
    page = max(0, min(page, total_pages - 1))

    start_idx = page * PAGE_SIZE
    page_items = filtered[start_idx : start_idx + PAGE_SIZE]

    keyboard = []
    # Add model buttons
    for name, path in page_items:
        prefix = "✅ " if current_model == path else "⬜ "
        cb_data = register_callback_path("model_set", path)
        keyboard.append([InlineKeyboardButton(f"{prefix}{name}", callback_data=cb_data)])

    # Add navigation row
    nav_row = []
    # Previous button
    if page > 0:
        q_cb = search_query[:15] if search_query else ""
        nav_row.append(InlineKeyboardButton("◀️ Previous", callback_data=f"model_p:{page-1}:{q_cb}"))
    
    # Page indicator
    nav_row.append(InlineKeyboardButton(f"Page {page+1}/{total_pages}", callback_data="model_noop"))

    # Next button
    if page < total_pages - 1:
        q_cb = search_query[:15] if search_query else ""
        nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"model_p:{page+1}:{q_cb}"))

    keyboard.append(nav_row)

    # Search & Manual Entry Row
    action_row = [
        InlineKeyboardButton("🔍 Search", callback_data="model_search"),
        InlineKeyboardButton("✍️ Manual Entry", callback_data="model_manual"),
    ]
    if search_query:
        action_row.append(InlineKeyboardButton("❌ Clear Search", callback_data="model_p:0:"))
    keyboard.append(action_row)

    # Main menu Back button
    keyboard.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_start")])

    # Text content
    text = "🤖 <b>Select Active Claude Model Override</b>:\n"
    if current_model:
        text += f"Current Active Model: <code>{current_model}</code>\n"
    else:
        text += "Current Active Model: <i>(None, using default)</i>\n"

    if search_query:
        text += f"Filter query: <code>{search_query}</code>\n"

    if not page_items:
        text += "\n⚠️ No models matching the query found."
    else:
        text += f"\nShowing {start_idx + 1} - {start_idx + len(page_items)} of {total_items} models."

    return text, InlineKeyboardMarkup(keyboard)


def make_settings_keyboard(
    web_tools_enabled: bool,
    debug_platform_edits: bool,
) -> Any:
    """Build a settings toggle keyboard showing current states."""
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return None

    web_status = "✅ Enabled" if web_tools_enabled else "❌ Disabled"
    debug_status = "✅ Enabled" if debug_platform_edits else "❌ Disabled"

    keyboard = [
        [
            InlineKeyboardButton(
                f"Web Search & Fetch: {web_status}",
                callback_data="settings_toggle:enable_web_server_tools",
            )
        ],
        [
            InlineKeyboardButton(
                f"Platform Debug Edits: {debug_status}",
                callback_data="settings_toggle:debug_platform_edits",
            )
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_start"),
            InlineKeyboardButton("❌ Close Settings", callback_data="settings_close"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def make_workspace_keyboard(workspace_dir: str, rel_path: str = "") -> tuple[str, Any]:
    """List directory contents and construct an inline keyboard directory explorer.

    Returns:
        A tuple of (text_content, InlineKeyboardMarkup)
    """
    import os
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return "", None

    # Sanitize and resolve target path
    base_abs = os.path.normpath(os.path.abspath(workspace_dir))
    target_abs = os.path.normpath(os.path.abspath(os.path.join(base_abs, rel_path)))

    try:
        common = os.path.commonpath([base_abs, target_abs])
        is_contained = os.path.normpath(common) == base_abs
    except Exception:
        is_contained = False

    if not is_contained:
        return "❌ Access Denied: Directory traversal blocked.", None

    if not os.path.exists(target_abs):
        return "❌ Path not found.", None

    if not os.path.isdir(target_abs):
        return "❌ Target is not a directory.", None

    # List files and directories
    try:
        entries = sorted(os.listdir(target_abs))
    except Exception as e:
        return f"❌ Failed to read directory: {e}", None

    keyboard = []

    # Back button if in a subdirectory
    if rel_path:
        parent_rel = os.path.dirname(rel_path)
        # Normalize relative path of parent
        parent_rel = "" if parent_rel in [".", "..", ""] else parent_rel.replace("\\", "/")
        cb_data = register_callback_path("workspace_ls", parent_rel) if parent_rel else "workspace_ls:"
        keyboard.append([InlineKeyboardButton("🔙 .. (Back)", callback_data=cb_data)])

    dirs = []
    files = []

    for name in entries:
        full_path = os.path.join(target_abs, name)
        item_rel = os.path.relpath(full_path, base_abs).replace("\\", "/")
        if os.path.isdir(full_path):
            dirs.append((name, item_rel))
        else:
            files.append((name, item_rel))

    # Add directories first
    for name, path in dirs[:15]:  # limit to prevent hitting button limit
        cb_data = register_callback_path("workspace_ls", path)
        keyboard.append([InlineKeyboardButton(f"📁 {name}", callback_data=cb_data)])

    # Add files
    for name, path in files[:15]:
        cb_data = register_callback_path("workspace_view", path)
        keyboard.append([InlineKeyboardButton(f"📄 {name}", callback_data=cb_data)])

    keyboard.append([
        InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_start"),
        InlineKeyboardButton("❌ Close Explorer", callback_data="settings_close")
    ])

    dir_name = os.path.basename(target_abs) or "Root"
    text = f"📁 <b>Workspace Explorer</b>: <code>{dir_name}</code>\n"
    if rel_path:
        text += f"Relative Path: <code>{rel_path}</code>\n"
    text += f"Total: {len(dirs)} folders, {len(files)} files."

    return text, InlineKeyboardMarkup(keyboard)
