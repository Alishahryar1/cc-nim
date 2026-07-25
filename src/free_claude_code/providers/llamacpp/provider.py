import os
from loguru import logger
import httpx

class LlamaCPPProvider:
    def __init__(self, base_url: str = None, default_model: str = "default"):
        raw_url = base_url or os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080")
        if not raw_url.endswith("/v1") and not raw_url.endswith("/v1/"):
            self.api_base = f"{raw_url.rstrip('/')}/v1"
        else:
            self.api_base = raw_url.rstrip('/')
        self.default_model = default_model

    async def execute(self, messages: list[dict], model: str = None, tools: list[dict] = None, stream: bool = False, **kwargs):
        target_model = model or self.default_model
        endpoint = f"{self.api_base}/chat/completions"
        
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        headers = {"Content-Type": "application/json"}
        logger.debug(f"Sending request to LlamaCPP provider at {endpoint} with model {target_model}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                async def response_generator():
                    async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            raise RuntimeError(f"LlamaCPP API error [{response.status_code}]: {error_text.decode('utf-8', errors='ignore')}")
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                yield line + "\n"
                return response_generator()
            else:
                response = await client.post(endpoint, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f"LlamaCPP API error [{response.status_code}]: {response.text}")
                return response.json()
