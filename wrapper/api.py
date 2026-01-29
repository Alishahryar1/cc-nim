from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from wrapper.cli_session import CLISession
from wrapper.parser import CLIParser
import os
import uvicorn
import logging
from dotenv import load_dotenv

# Load .env from the root directory
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration from environment
workspace_env = os.getenv("CLAUDE_WORKSPACE", "agent_workspace")
WORKSPACE = os.path.normpath(os.path.abspath(workspace_env))
print(f"--- WRAPPER CONFIG ---", flush=True)
print(f"WORKSPACE: {WORKSPACE}", flush=True)

# The proxy server is running on port 8082 based on user context
API_URL = os.getenv("ANTHROPIC_API_URL", "http://localhost:8082/v1")
ALLOWED_DIRS = os.getenv("ALLOWED_DIRS", "").split(",")
ALLOWED_DIRS = [os.path.normpath(d.strip()) for d in ALLOWED_DIRS if d.strip()]
print(f"ALLOWED_DIRS: {ALLOWED_DIRS}", flush=True)

session = CLISession(WORKSPACE, API_URL, ALLOWED_DIRS)


@app.post("/stop")
async def stop_session():
    """Forcefully terminates the current Claude CLI process."""
    if session.process and session.process.returncode is None:
        logger.info("Forcefully terminating Claude process...")
        try:
            session.process.terminate()
            return {"status": "terminated"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "no_active_process"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Bot connected to WebSocket")

    try:
        while True:
            # Receive task from bot
            data = await websocket.receive_json()
            task = data.get("task")

            if not task:
                continue

            logger.info(f"Received task: {task}")

            # Start Claude session and stream events back
            async for event in session.start_task(task):
                parsed = CLIParser.parse_event(event)
                if parsed:
                    await websocket.send_json(parsed)
                else:
                    # Send raw for debugging if it's unknown but relevant
                    if event.get("type") in ["system", "result"]:
                        await websocket.send_json(
                            {"type": "status", "data": event.get("subtype") or "update"}
                        )

    except WebSocketDisconnect:
        logger.info("Bot disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
