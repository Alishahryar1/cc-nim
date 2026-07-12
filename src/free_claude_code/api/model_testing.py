import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import httpx
from pydantic import BaseModel

class ModelTestResult(BaseModel):
    model: str
    status: str  # "pending", "running", "passed", "failed", "timeout", "error"
    http_status: int = 0
    latency_ms: int = 0
    response: str = ""
    error_type: str = ""
    error_message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

class ModelTestingManager:
    def __init__(self):
        self.is_running = False
        self.total_models = 0
        self.tested_count = 0
        self.results: Dict[str, ModelTestResult] = {}
        self.start_time = 0.0
        self.elapsed_seconds = 0.0
        self._current_task = None
        self._save_in_progress = False
        self._save_needed = False

    def start_test(self, model_ids: List[str], auth_token: str, port: int) -> bool:
        if self.is_running:
            return False
        self.is_running = True
        self.total_models = len(model_ids)
        self.tested_count = 0
        self.start_time = time.time()
        self.elapsed_seconds = 0.0
        self.results = {
            m_id: ModelTestResult(model=m_id, status="pending")
            for m_id in model_ids
        }
        # Run in background via asyncio
        task = asyncio.create_task(self._run_tests(model_ids, auth_token, port))
        self._current_task = task
        
        def _on_done(t):
            self._current_task = None
            
        task.add_done_callback(_on_done)
        return True

    async def request_save(self):
        if self._save_in_progress:
            self._save_needed = True
            return
        self._save_in_progress = True
        self._save_needed = False
        try:
            await asyncio.to_thread(self.save_results_files)
        finally:
            self._save_in_progress = False
            if self._save_needed:
                asyncio.create_task(self.request_save())

    async def _run_tests(self, model_ids: List[str], auth_token: str, port: int):
        headers = {
            "anthropic-version": "2023-06-01",
        }
        if auth_token:
            headers["x-api-key"] = auth_token

        base_url = f"http://127.0.0.1:{port}"
        semaphore = asyncio.Semaphore(3)

        async def test_single_model(model_id: str):
            async with semaphore:
                self.results[model_id].status = "running"
                body = {
                    "model": model_id,
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                }
                t0 = time.time()
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        r = await client.post(
                            f"{base_url}/v1/messages",
                            headers=headers,
                            json=body
                        )
                    latency = round((time.time() - t0) * 1000)
                    self.results[model_id].latency_ms = latency
                    self.results[model_id].http_status = r.status_code

                    if r.status_code == 200:
                        self.results[model_id].status = "passed"
                        resp_text = ""
                        in_tok = 0
                        out_tok = 0
                        is_sse = "event-stream" in r.headers.get("Content-Type", "").lower()
                        text_text = ""
                        thinking_text = ""

                        if is_sse:
                            for line in r.text.split("\n"):
                                line = line.strip()
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        continue
                                    try:
                                        evt = json.loads(data_str)
                                        etype = evt.get("type", "")
                                        if etype == "content_block_delta":
                                            delta = evt.get("delta", {})
                                            if delta.get("type") == "text_delta":
                                                text_text += delta.get("text", "")
                                            elif delta.get("type") == "thinking_delta":
                                                thinking_text += delta.get("thinking", "")
                                        elif etype == "content_block_start":
                                            block = evt.get("content_block", {})
                                            if block.get("type") == "text":
                                                text_text += block.get("text", "")
                                            elif block.get("type") == "thinking":
                                                thinking_text += block.get("thinking", "")
                                        elif etype == "message_start":
                                            msg = evt.get("message", {})
                                            usage = msg.get("usage", {})
                                            if usage:
                                                in_tok = usage.get("input_tokens", 0)
                                                out_tok = usage.get("output_tokens", 0)
                                        elif etype == "message_delta":
                                            usage = evt.get("usage", {})
                                            if usage:
                                                if usage.get("input_tokens"):
                                                    in_tok = usage.get("input_tokens")
                                                if usage.get("output_tokens"):
                                                    out_tok = usage.get("output_tokens")
                                    except Exception:
                                        pass
                        else:
                            try:
                                data = r.json()
                                for block in data.get("content", []):
                                    if block.get("type") == "text":
                                        text_text += block.get("text", "")
                                    elif block.get("type") == "thinking":
                                        thinking_text += block.get("thinking", "")
                                usage = data.get("usage", {})
                                in_tok = usage.get("input_tokens", 0)
                                out_tok = usage.get("output_tokens", 0)
                            except Exception:
                                pass

                        resp_text = text_text if text_text.strip() else thinking_text
                        self.results[model_id].response = resp_text
                        self.results[model_id].input_tokens = in_tok
                        self.results[model_id].output_tokens = out_tok
                    else:
                        self.results[model_id].status = "failed"
                        err_msg = r.text
                        err_type = "http_error"
                        try:
                            err_data = r.json()
                            err_obj = err_data.get("error", {})
                            err_type = err_obj.get("type", "")
                            err_msg = err_obj.get("message", "")
                        except Exception:
                            pass
                        self.results[model_id].error_type = err_type
                        self.results[model_id].error_message = err_msg
                except httpx.TimeoutException:
                    self.results[model_id].status = "timeout"
                    self.results[model_id].error_message = "Request timed out after 15s"
                except Exception as e:
                    self.results[model_id].status = "error"
                    self.results[model_id].error_message = str(e)
                finally:
                    self.tested_count += 1
                    self.elapsed_seconds = round(time.time() - self.start_time, 1)
                    await self.request_save()

        await asyncio.gather(*(test_single_model(m_id) for m_id in model_ids))
        self.is_running = False
        await self.request_save()

    def save_results_files(self):
        # Build working and failed lists
        working = []
        failed = []
        for model_id, res in self.results.items():
            if res.status == "passed":
                working.append({
                    "model": res.model,
                    "status": res.http_status,
                    "latency_ms": res.latency_ms,
                    "response": res.response,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens
                })
            elif res.status in ["failed", "timeout", "error"]:
                failed.append({
                    "model": res.model,
                    "status": res.http_status,
                    "latency_ms": res.latency_ms,
                    "error_type": res.error_type or res.status,
                    "error_message": res.error_message
                })

        working.sort(key=lambda x: x["latency_ms"])

        # Output folder
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            (output_dir / "working.json").write_text(
                json.dumps(working, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (output_dir / "failed.json").write_text(
                json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            js_content = f"""window.modelResults = {{
  total: {self.total_models},
  tested: {self.tested_count},
  working: {json.dumps(working, ensure_ascii=False)},
  failed: {json.dumps(failed, ensure_ascii=False)},
  lastUpdated: "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
}};"""
            (output_dir / "results_data.js").write_text(js_content, encoding="utf-8")
        except Exception:
            pass


# Singleton instance
testing_manager = ModelTestingManager()
