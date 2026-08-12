# How to create and register a custom API handler

This guide walks you through creating a custom request handler in `src/free_claude_code/api/handlers/` and registering its HTTP route in the FastAPI server router (`routes.py`).

## Prerequisites

- Free Claude Code codebase installed and operational in your Python environment.
- Familiarity with FastAPI dependency injection (`Depends`) and Pydantic request models.

## Steps

### 1. Create the handler implementation

Create a new file `src/free_claude_code/api/handlers/custom_feature.py` to encapsulate your handler logic:

```python
from free_claude_code.application.ports import ProviderResolver
from free_claude_code.config.settings import Settings


class CustomFeatureHandler:
    """Processes requests for the custom API feature."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
    ) -> None:
        self.settings = settings
        self.provider_resolver = provider_resolver

    async def handle_request(self, payload: dict, request_id: str) -> dict:
        """Execute feature processing and return response payload."""
        return {
            "status": "success",
            "request_id": request_id,
            "processed_keys": list(payload.keys()),
        }
```

### 2. Export the handler in `handlers/__init__.py`

Edit `src/free_claude_code/api/handlers/__init__.py` to re-export your new handler class:

```python
from .custom_feature import CustomFeatureHandler

__all__ = [
    "MessagesHandler",
    "ResponsesHandler",
    "TokenCountHandler",
    "CustomFeatureHandler",
]
```

### 3. Register the endpoint in `routes.py`

Open `src/free_claude_code/api/routes.py` and register the route using `APIRouter` with runtime lease acquisition and authentication dependency:

```python
from fastapi import Depends, Request
from free_claude_code.api.dependencies import (
    get_services,
    require_proxy_auth,
    resolve_provider,
)
from free_claude_code.api.handlers import CustomFeatureHandler
from free_claude_code.api.ports import ApiServices
from free_claude_code.api.request_ids import get_request_id


@router.post("/v1/custom-feature")
async def create_custom_feature(
    request: Request,
    payload: dict,
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_proxy_auth),
):
    """Custom endpoint execution with runtime lease safety."""
    lease = await services.requests.acquire()
    try:
        handler = CustomFeatureHandler(
            settings=lease.settings,
            provider_resolver=lambda provider_type: resolve_provider(
                provider_type, lease=lease
            ),
        )
        return await handler.handle_request(payload, request_id=get_request_id(request))
    finally:
        await lease.release()
```

## Verify

### 1. Execute static checks and tests

Run `uv run pytest` targeting unit tests in `tests/unit/api/` to verify route resolution:

```bash
uv run pytest tests/unit/api/ -v
```

### 2. Test endpoint with `curl`

Start the local proxy server (`fcc-server`) and send a test request:

```bash
curl -X POST http://127.0.0.1:8000/v1/custom-feature \
  -H "Content-Type: application/json" \
  -d '{"data": "test_payload"}'
```

Verify that the response returns `HTTP 200 OK` with JSON content:

```json
{
  "status": "success",
  "request_id": "...-...",
  "processed_keys": ["data"]
}
```
