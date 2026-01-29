import os
import json
import asyncio
import logging
import websockets
from dotenv import load_dotenv
from telethon import TelegramClient, events

# Load .env from the root directory of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

# Configuration
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
ALLOWED_USER_ID = os.getenv("ALLOWED_TELEGRAM_USER_ID")
WRAPPER_WS_URL = os.getenv("WRAPPER_WS_URL", "ws://localhost:8083/ws")

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Telethon Client
# Store the session file in the project root to avoid duplicates
session_path = os.path.join(project_root, "claude_bot.session")
print(f"DEBUG: Using session file at: {session_path}", flush=True)

client = TelegramClient(
    os.path.join(project_root, "claude_bot"), int(API_ID) if API_ID else 0, API_HASH
)


# Global state to cache the owner info
my_info = None


@client.on(events.NewMessage())
async def handle_new_message(event):
    global my_info
    if not my_info:
        my_info = await client.get_me()

    # Handle /stop command
    if event.text == "/stop":
        import httpx

        try:
            # Derive HTTP URL from WS URL
            http_url = (
                WRAPPER_WS_URL.replace("ws://", "http://")
                .replace("wss://", "https://")
                .replace("/ws", "/stop")
            )
            async with httpx.AsyncClient() as hclient:
                r = await hclient.post(http_url)
                await event.reply(
                    f"⏹ **Stop signal sent.**\nResult: `{r.json().get('status')}`"
                )
        except Exception as e:
            await event.reply(f"❌ Failed to send stop signal: {e}")
        return

    # DEBUG: Simple message detection
    print(
        f"DEBUG: Message detected in chat {event.chat_id}: '{event.text[:30]}...'",
        flush=True,
    )

    sender = await event.get_sender()
    if not sender:
        return

    user_id = str(sender.id)

    # Is it me or the authorized user?
    is_me = user_id == str(my_info.id)
    is_authorized = ALLOWED_USER_ID and user_id == str(ALLOWED_USER_ID)

    print(
        f"DEBUG: from_id={user_id} | is_me={is_me} | is_authorized={is_authorized}",
        flush=True,
    )

    # Security check
    if not is_me and not is_authorized:
        return

    # Skip empty, commands, or our own bot tokens
    if (
        not event.text
        or event.text.startswith("/")
        or any(event.text.startswith(p) for p in ["🤖", "🔧", "✅", "❌"])
    ):
        return

    print(f"DEBUG: ACCEPTED task from {user_id}: {event.text}", flush=True)
    logger.info(f"Accepted task from {user_id}: {event.text}")

    # Send initial status
    status_msg = await event.reply("🧠 **Claude is thinking...**")

    current_content = ""
    last_update_time = 0
    update_interval = 1.0  # Update every 1 second to avoid rate limits

    async def update_telegram_ui(text, status_prefix=""):
        nonlocal last_update_time
        import time

        now = time.time()
        if now - last_update_time < update_interval:
            return

        try:
            # Format: [Status] + Content
            display_text = f"{status_prefix}\n\n{text}" if status_prefix else text
            # Telegram limit check (simple)
            if len(display_text) > 4000:
                display_text = display_text[-4000:]  # Show last 4k chars

            await status_msg.edit(display_text, parse_mode="markdown")
            last_update_time = now
        except Exception as ui_e:
            logger.debug(f"UI Update failed: {ui_e}")

    try:
        async with websockets.connect(WRAPPER_WS_URL) as ws:
            await ws.send(json.dumps({"task": event.text}))

            async for message in ws:
                data = json.loads(message)
                etype = data.get("type")

                if etype == "content":
                    chunk = data.get("text", "")
                    current_content += chunk
                    await update_telegram_ui(
                        current_content, "🧠 **Claude is writing...**"
                    )

                elif etype == "tool_start":
                    tools = data.get("tools", [])
                    tool_names = [t.get("name") for t in tools]
                    await update_telegram_ui(
                        current_content, f"🔧 **Running:** `{', '.join(tool_names)}`"
                    )

                elif etype == "complete":
                    # Final update
                    final_text = (
                        current_content
                        if current_content
                        else "Task finished with no output."
                    )
                    await status_msg.edit(
                        f"✅ **Task Complete**\n\n{final_text}", parse_mode="markdown"
                    )
                    break

                elif etype == "error":
                    await event.reply(f"❌ **Claude Error:**\n{data.get('message')}")
                    break

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"DEBUG: Exception during WebSocket communication: {e}", flush=True)
        await event.reply(
            f"💥 **Connection failed:** {e}\n_Is the wrapper running on port 8083?_"
        )


async def main():
    print("Starting Telegram Client...", flush=True)
    await client.start()

    # Get and cache owner info
    global my_info
    my_info = await client.get_me()

    print(f"--- BOT STARTED ---", flush=True)
    print(
        f"Logged in as: {my_info.first_name} (@{my_info.username if my_info.username else 'No Username'})",
        flush=True,
    )
    print(f"Your User ID is: {my_info.id}", flush=True)

    # TEST: Send a message to Saved Messages to see if we're alive
    try:
        await client.send_message(
            "me",
            "🚀 **Claude Agent is now online!**\nSend me a task right here in Saved Messages to begin.",
        )
        print("DEBUG: Sent startup message to Saved Messages ('me').", flush=True)
    except Exception as e:
        print(f"DEBUG: Failed to send startup message: {e}", flush=True)

    print("Client is running. Listening for ALL messages...", flush=True)
    await client.run_until_disconnected()


if __name__ == "__main__":
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not set in .env")
        exit(1)

    asyncio.run(main())
