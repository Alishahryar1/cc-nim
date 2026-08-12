# Free Claude Code (FCC) - Windows User Guide (`omni01-Cell` Fork)

This document is the official Windows guide for the **[omni01-Cell/free-claude-code](https://github.com/omni01-Cell/free-claude-code)** fork. It covers installation, usage, development, and configuration for all standard and custom-added providers and features.

---

## 1. Fork Overview & Custom Capabilities

This fork enhances standard Free Claude Code with:
- **Google Antigravity CLI Provider**: Full OAuth 2.0 PKCE Connected Account support, 1M token context window (`1,048,576` tokens), multimodal image support, and real-time SSE stream tool deduplication (`seen_tool_calls` & `active_tool_by_name`).
- **AgentRouter Provider**: Support for `claude-opus-5`, anti-401 header impersonation, and granular reasoning effort mapping (`low`, `medium`, `high`, `max`, `ultra`).
- **Additional Custom Providers**: CommandCode, TokenRouter, Alibaba DashScope, OpenAI Compatible, and Anthropic Compatible.
- **`fcc-codex-desktop` Launcher**: Hybrid ephemeral/persistent Windows launcher for Codex Desktop with automatic TOML path escaping.
- **PowerShell Automation**: Native installation (`scripts/install.ps1`) and CI verification (`scripts/ci.ps1`) scripts for Windows.

---

## 2. System Requirements

- **Operating System**: Windows 10 or Windows 11 (x64 or ARM64 via x64 emulation).
- **PowerShell**: PowerShell 5.1+ or PowerShell Core (7+).
- **Git**: [Git for Windows](https://git-scm.com/download/win) (recommended).
- **Python**: Python 3.14 (managed automatically via `uv`).

---

## 3. Installation & Updates

To install or update the `omni01-Cell` fork on Windows, run the following command in PowerShell:

```powershell
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/omni01-Cell/free-claude-code/main/scripts/install.ps1")))
```

### What `install.ps1` Handles:
1. Installs/updates `uv` on Windows.
2. Installs or upgrades `free-claude-code` from the `omni01-Cell` repository.
3. Creates Windows Start Menu shortcuts (`Free Claude Code.lnk`) and Desktop icons.
4. Auto-initializes `%USERPROFILE%\.claude.json` with the FCC proxy local endpoint (`http://127.0.0.1:8082/v1`).

---

## 4. Running Free Claude Code on Windows

### Option A: Windows System Tray (Recommended)
1. Launch **Free Claude Code** from the Start Menu or Desktop shortcut.
2. FCC runs in the background and places an icon in the Windows System Tray.
3. **Left-click**: Opens the **Admin UI** (`http://127.0.0.1:8082/admin`).
4. **Right-click**: Access menu options (Status, Admin, Restart, Quit).

### Option B: Terminal / Console Mode
To run the server manually in PowerShell or Command Prompt:

```powershell
fcc-server
```

---

## 5. Custom Providers & Configuration (Admin UI)

Access the Admin UI at [http://127.0.0.1:8082/admin](http://127.0.0.1:8082/admin) to set up any of the custom providers:

### 🌌 5.1. Google Antigravity CLI (`antigravity`)
- **Auth Kind**: `CONNECTED_ACCOUNT` (OAuth 2.0 PKCE loopback server).
- **Setup**: In Admin UI, click **Connect Account**. A browser tab opens for Google OAuth authorization.
- **Zero-Config Discovery**: Automatically detects local tokens in `%USERPROFILE%\.gemini\antigravity-cli\antigravity-oauth-token`.
- **Features**: 1,048,576 token context window, image input processing, and SSE tool call deduplication.

### 🔀 5.2. AgentRouter (`agentrouter`)
- **Default Base URL**: `https://ps.air-outer.com/v1`
- **Environment Variable**: `AGENTROUTER_API_KEY`
- **Reasoning Effort Levels**:
  - `low` -> 1,000 thinking tokens
  - `medium` -> 5,000 thinking tokens
  - `high` -> 10,000 thinking tokens
  - `max` -> 20,000 thinking tokens
  - `ultra` -> 32,000 thinking tokens

### ⚡ 5.3. CommandCode (`commandcode`)
- **Default Base URL**: `https://api.commandcode.ai/provider/v1`
- **Environment Variable**: `COMMANDCODE_API_KEY`

### 🎟️ 5.4. TokenRouter (`tokenrouter`)
- **Default Base URL**: `https://api.tokenrouter.com/v1`
- **Environment Variable**: `TOKENROUTER_API_KEY`

### ☁️ 5.5. Alibaba DashScope (`alibaba`)
- **Default Base URL**: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **Environment Variable**: `ALIBABA_API_KEY`

### 🔗 5.6. OpenAI Compatible (`openai_compatible`) & Anthropic Compatible (`anthropic_compatible`)
- Custom base URLs and API key configuration for any third-party gateway or self-hosted endpoint.

---

## 6. Client Integration Guide

### 💻 6.1. Claude Code CLI
Once installed, open PowerShell and run:

```powershell
claude
```

### 🤖 6.2. Codex Desktop Launcher (`fcc-codex-desktop`)

This fork includes a custom Windows launcher for Codex Desktop:

```powershell
fcc-codex-desktop
```

#### Launcher Features:
- **Automatic Path Resolution**: Finds `Codex.exe` or `ChatGPT.exe` under `%LOCALAPPDATA%` or system PATH.
- **Ephemeral Mode**: Automatically injects `model_provider = "fcc"` into `%USERPROFILE%\.codex\config.toml` upon launch and cleans up on exit.
- **Persistent Setup**: Run `fcc-codex-desktop --setup` to permanently configure Codex Desktop.
- **Reset / Restore**: Run `fcc-codex-desktop --reset` to restore your original config file.
- **Windows Path Escaping**: Automatically escapes Windows paths (`C:\Users\...`) to prevent TOML parsing errors.

### 🔌 6.3. VS Code (Codex Extension)
Edit `%USERPROFILE%\.codex\config.toml`:

```toml
model_catalog_json = "C:/Users/YOUR_USERNAME/.fcc/codex-model-catalog.json"
model_provider = "fcc"
model = "antigravity/gemini-2.5-pro"

[model_providers.fcc]
name = "Free Claude Code"
base_url = "http://127.0.0.1:8082/v1"
wire_api = "responses"

[model_providers.fcc.auth]
command = "fcc-codex"
args = ["--print-proxy-auth-token"]
```

---

## 7. Windows Environment & Path Reference

| Item | Windows Location |
| :--- | :--- |
| **FCC Core Configuration** | `%USERPROFILE%\.fcc\` |
| **Claude Code Profile** | `%USERPROFILE%\.claude.json` |
| **Codex Configuration** | `%USERPROFILE%\.codex\config.toml` |
| **Codex Model Catalog** | `%USERPROFILE%\.fcc\codex-model-catalog.json` |
| **Google Antigravity OAuth Tokens** | `%USERPROFILE%\.gemini\antigravity-cli\` |
| **Start Menu Shortcut** | `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Free Claude Code.lnk` |
| **JetBrains ACP Manifest** | `%APPDATA%\JetBrains\acp-agents\installed.json` |

---

## 8. Development & CI Verification on Windows

To run full validation on Windows using PowerShell:

```powershell
.\scripts\ci.ps1
```

The script runs 5 strict checks:
1. `suppression-grep` - Check for forbidden ignores
2. `ruff-format` - Code formatting (`uv run ruff format`)
3. `ruff-check` - Linting (`uv run ruff check`)
4. `ty-check` - Type safety (`uv run ty check`)
5. `pytest` - Test suite (`uv run pytest`)

Run a specific job:
```powershell
.\scripts\ci.ps1 -Only pytest
```

---

## 9. Uninstallation

To uninstall Free Claude Code from Windows:

```powershell
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/omni01-Cell/free-claude-code/main/scripts/uninstall.ps1")))
```
