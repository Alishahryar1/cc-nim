# 🤖 Free Claude Code Setup Guide (OpenRouter GPT-OSS)

Follow these step-by-step instructions to set up **Free Claude Code** on your system using OpenRouter's free GPT-OSS model with the Claude Code VS Code extension.

---

### 1. Install `uv` (Python Package & Environment Manager)
Open a **PowerShell** terminal and run the following command to install `uv` and Python 3.14 on your Windows machine:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "C:\Users\$env:USERNAME\.local\bin;$env:Path"
uv self update
uv python install 3.14
```

---

### 2. Clone and Configure the Repository
1. Open your terminal, clone this repository, and navigate inside:
   ```cmd
   git clone https://github.com/Alishahryar1/free-claude-code.git
   cd free-claude-code
   ```
2. Copy the template configuration file to create your active environment file:
   ```cmd
   copy .env.example .env
   ```

---

### 3. Configure the `.env` File
Open the newly created `.env` file in a text editor and update the following settings:

1. **OpenRouter API Key** (replace with your active OpenRouter key):
   ```dotenv
   OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```

2. **Model Tier Routing** (maps all Claude tiers to the free GPT-OSS-120B model):
   ```dotenv
   MODEL_OPUS="open_router/openai/gpt-oss-120b:free"
   MODEL_SONNET="open_router/openai/gpt-oss-120b:free"
   MODEL_HAIKU="open_router/openai/gpt-oss-120b:free"
   MODEL="open_router/openai/gpt-oss-120b:free"
   ```

3. **Disable Model Thinking** (crucial to prevent the Claude Code parser from crashing on unsupported thinking block events):
   ```dotenv
   ENABLE_MODEL_THINKING=false
   ```

4. **Security Auth Token** (leave as default):
   ```dotenv
   ANTHROPIC_AUTH_TOKEN="freecc"
   ```

---

### 4. Configure VS Code Settings
To point the Claude Code VS Code extension at your local proxy:
1. Open **VS Code**.
2. Press **Ctrl + ,** to open your Settings.
3. Search for: `claude-code.environmentVariables`.
4. Click **Edit in settings.json** and add the following entry inside the root JSON brackets:
   ```json
   "claudeCode.environmentVariables": [
       { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
       { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" },
       { "name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "value": "1" }
   ]
   ```
5. Press **Ctrl + Shift + P**, search for **Developer: Reload Window**, and reload your editor.

---

### 5. Start the Proxy Server
Whenever you want to use the extension, open a terminal inside the cloned `free-claude-code` directory and start the server:
```cmd
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

Once the terminal outputs `Uvicorn running on http://0.0.0.0:8082`, you are ready to write code using Claude Code for free!
