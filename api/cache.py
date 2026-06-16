import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional
from loguru import logger

class ResponseCache:
    def __init__(self, cache_dir: str = ".fcc_cache", enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, request_data: Any) -> str:
        # Create a stable hash of the request data
        if hasattr(request_data, "model_dump_json"):
            # Exclude fields that might vary but don't change the semantic request if needed
            # For now dump everything but ensure it's sorted
            data_str = request_data.model_dump_json()
        elif isinstance(request_data, dict):
            data_str = json.dumps(request_data, sort_keys=True)
        else:
            data_str = str(request_data)

        key = hashlib.sha256(data_str.encode()).hexdigest()
        return key

    def get(self, request_data: Any) -> Optional[list[str]]:
        if not self.enabled:
            return None
        key = self._get_key(request_data)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    logger.debug("Cache hit for request {}", key)
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to read cache file {}: {}", cache_file, e)
        else:
            logger.debug("Cache miss for request {}", key)
        return None

    def set(self, request_data: Any, response_chunks: list[str]) -> None:
        if not self.enabled:
            return
        key = self._get_key(request_data)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(response_chunks, f)
            logger.debug("Cached response for request {}", key)
        except Exception as e:
            logger.warning("Failed to write cache file {}: {}", cache_file, e)
