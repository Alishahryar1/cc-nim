import asyncio
import json
from typing import Any, AsyncIterator, List
from loguru import logger
from api.models.anthropic import MessagesRequest, Message, ContentBlockText, ContentBlockImage
from core.anthropic import SSEBuilder

class VisionBridge:
    def __init__(self, proxy_service: Any):
        self.proxy_service = proxy_service

    def has_images(self, request_data: MessagesRequest) -> bool:
        """True if the request contains image content blocks."""
        for msg in request_data.messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if getattr(block, "type", None) == "image" or (isinstance(block, dict) and block.get("type") == "image"):
                        return True
        return False

    async def bridge_vision(self, request_data: MessagesRequest, target_model_ref: str) -> AsyncIterator[str]:
        """Convert images to text descriptions using a Vision Expert, then call target model."""
        logger.info("Vision Bridge triggered for model: {}", target_model_ref)

        # Pick the best vision expert from ALL providers
        vision_expert = self._select_best_vision_expert()

        # 1. Process all messages to extract and describe images
        new_messages = []
        for msg in request_data.messages:
            if isinstance(msg.content, str):
                new_messages.append(msg)
                continue

            new_content = []
            for block in msg.content:
                is_image = getattr(block, "type", None) == "image" or (isinstance(block, dict) and block.get("type") == "image")
                if is_image:
                    # Call Vision Expert to describe this image
                    description = await self._describe_image(vision_expert, block)
                    new_content.append(ContentBlockText(type="text", text=f"\n[IMAGE DESCRIPTION]: {description}\n"))
                else:
                    new_content.append(block)

            new_messages.append(Message(role=msg.role, content=new_content))

        # 2. Call target model with augmented text context
        bridged_request = request_data.model_copy(update={"messages": new_messages})

        candidates = self.proxy_service._model_router.resolve_candidates(target_model_ref)
        return self.proxy_service._stream_with_failover(bridged_request, candidates)

    def _select_best_vision_expert(self) -> str:
        """Dynamically pick the best vision model based on performance and availability."""
        candidates = [
            self.proxy_service._settings.vision_expert_model,
            self.proxy_service._settings.model_opus,
            "nvidia_nim/nvidia/llama-3.2-90b-vision-instruct",
            "open_router/google/gemini-2.0-flash-001",
            "gemini/gemini-1.5-flash",
        ]

        # Filter None and duplicates
        models = []
        seen = set()
        for m in candidates:
            if m and m not in seen:
                models.append(m)
                seen.add(m)

        if not models: return "nvidia_nim/nvidia/llama-3.2-90b-vision-instruct"

        # Re-use scoring logic
        def _score(m_ref: str) -> float:
            try:
                p_id = m_ref.split("/", 1)[0]
                from api.performance import performance_tracker
                metrics = performance_tracker.get_metrics(p_id)
                if metrics.success_count == 0: return 10.0
                return (1.0 - metrics.error_rate) / max(metrics.avg_latency, 0.001)
            except:
                return 0.0

        models.sort(key=_score, reverse=True)
        return models[0]

    async def _describe_image(self, vision_model: str, image_block: Any) -> str:
        """Get a text description for a single image block."""
        logger.debug("Generating image description using {}", vision_model)

        # Build a single-turn request for the vision model
        desc_request = MessagesRequest(
            model=vision_model,
            messages=[
                Message(
                    role="user",
                    content=[
                        image_block,
                        ContentBlockText(type="text", text="Describe this image in detail for a text-only AI model. Focus on layout, text, and key visual elements.")
                    ]
                )
            ],
            stream=False
        )

        try:
            candidates = self.proxy_service._model_router.resolve_candidates(vision_model)
            resolved = candidates[0]
            provider = self.proxy_service._provider_getter(resolved.provider_id)

            routed = desc_request.model_copy(deep=True)
            routed.model = resolved.provider_model

            full_text = []
            async for chunk in provider.stream_response(routed):
                for line in chunk.splitlines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]": continue
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text":
                                    full_text.append(delta.get("text", ""))
                            elif data.get("type") == "content_block_start":
                                block = data.get("content_block", {})
                                if block.get("type") == "text":
                                    full_text.append(block.get("text", ""))
                        except:
                            pass
            return "".join(full_text)
        except Exception as e:
            logger.error("Vision Bridge: Description failed: {}", e)
            return "[Error describing image]"
