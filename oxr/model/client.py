import asyncio
import base64
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from oxr.config.settings import OXRModelConfig, settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
}


class ModelOutput(str):
    """String-compatible model output with completion metadata attached."""

    finish_reason: str
    usage: Any

    def __new__(
        cls,
        content: str,
        *,
        finish_reason: str = "unknown",
        usage: Any = None,
    ) -> "ModelOutput":
        instance = super().__new__(cls, content)
        instance.finish_reason = finish_reason
        instance.usage = usage
        return instance


def resolve_model_options(model_options: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    resolved = dict(DEFAULT_MODEL_OPTIONS)
    if model_options:
        resolved.update({
            key: float(value)
            for key, value in model_options.items()
            if key in resolved and value is not None
        })
    return resolved


class OXRModelClient:
    """Persistent OpenAI-compatible client used only for OXR requests."""

    def __init__(self, config: Optional[OXRModelConfig] = None):
        self._config = config
        self._client: Optional[AsyncOpenAI] = None
        self._request_loop: Optional[asyncio.AbstractEventLoop] = None
        self._request_limit: Optional[int] = None
        self._request_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def config(self) -> OXRModelConfig:
        return self._config or settings.oxr_model

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        if self._client is None and self.config.url:
            self._client = AsyncOpenAI(
                base_url=self.config.url,
                api_key="EMPTY",
                max_retries=0,
                http_client=DefaultAsyncHttpxClient(trust_env=False),
            )
        return self._client

    def get_request_semaphore(self) -> asyncio.Semaphore:
        """Return the process client's shared limiter for the current event loop."""
        loop = asyncio.get_running_loop()
        limit = self.config.max_concurrency
        if (
            self._request_loop is not loop
            or self._request_limit != limit
            or self._request_semaphore is None
        ):
            self._request_loop = loop
            self._request_limit = limit
            self._request_semaphore = asyncio.Semaphore(limit)
        assert self._request_semaphore is not None
        return self._request_semaphore

    async def aclose(self) -> None:
        try:
            if self._client is not None:
                await self._client.close()
        finally:
            self._client = None
            self._request_loop = None
            self._request_limit = None
            self._request_semaphore = None

    def _encode_image(self, image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Unable to encode image as PNG.")
        return base64.b64encode(buffer).decode("utf-8")

    def _image_to_data_url(self, image: np.ndarray) -> str:
        return f"data:image/png;base64,{self._encode_image(image)}"

    async def _call_vision_model(
        self,
        prompt: str,
        image: np.ndarray,
        timeout: int,
        model_options: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.client:
            raise RuntimeError("OXR model URL is not configured.")
        if image is None:
            raise ValueError("An image is required for vision model calls.")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(image)},
            },
            {"type": "text", "text": prompt},
        ]
        resolved_options = resolve_model_options(model_options)

        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": content}],
                timeout=timeout,
                temperature=resolved_options["temperature"],
                top_p=resolved_options["top_p"],
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                extra_body={
                    "repetition_penalty": resolved_options["repetition_penalty"],
                },
            ),
            timeout=timeout,
        )
        choice = response.choices[0]
        return ModelOutput(
            choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", None) or "unknown",
            usage=getattr(response, "usage", None),
        )

    async def request_reading_order(
        self,
        prompt: str,
        image: np.ndarray,
        max_tokens: int,
    ) -> str:
        """Submit a deterministic OXR reading-order request with a prepared prompt."""
        return await self._call_vision_model(
            prompt=prompt,
            image=image,
            timeout=self.config.timeout,
            model_options=None,
            max_tokens=max_tokens,
        )

    async def recognize_text(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        from oxr.model.prompts import PROMPT_TEXT_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_TEXT_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_formula(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        from oxr.model.prompts import PROMPT_FORMULA_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_FORMULA_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_table(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        from oxr.model.prompts import PROMPT_TABLE_RECOGNITION
        from oxr.utils.table_image import (
            merge_table_chunk_outputs,
            prepare_table_image,
            split_long_table_image,
        )
        from oxr.utils.table import keep_complete_html_tables

        async def recognize_part(part: np.ndarray) -> ModelOutput:
            prepared = prepare_table_image(part)
            async with self.get_request_semaphore():
                output = await self._call_vision_model(
                    prompt=PROMPT_TABLE_RECOGNITION,
                    image=prepared,
                    timeout=self.config.timeout,
                    model_options=model_options,
                )
            cleaned = keep_complete_html_tables(str(output))
            if cleaned != output:
                logger.warning("Removed text outside the completed HTML table output.")
                return ModelOutput(
                    cleaned,
                    finish_reason=output.finish_reason,
                    usage=output.usage,
                )
            return output

        chunks = split_long_table_image(image)
        if len(chunks) == 1:
            return await recognize_part(chunks[0])

        tasks = [asyncio.create_task(recognize_part(chunk)) for chunk in chunks]
        try:
            outputs = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        merged = merge_table_chunk_outputs([str(output) for output in outputs])
        if merged is None:
            logger.warning(
                "Could not safely merge %s table chunks because their repeated rows differed; "
                "retrying the original table image.",
                len(chunks),
            )
            return await recognize_part(image)

        merged_html, merged_rows = merged
        logger.info(
            "Recognized long table in %s row chunks; assembled %s HTML rows in source order.",
            len(chunks),
            merged_rows,
        )
        finish_reason = (
            "length"
            if any(output.finish_reason == "length" for output in outputs)
            else "stop"
        )
        return ModelOutput(merged_html, finish_reason=finish_reason)

    async def recognize_code(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        from oxr.model.prompts import PROMPT_CODE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_CODE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def analyze_layout(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        """Run the diagnostic OXR layout prompt outside the document pipeline."""
        from oxr.model.prompts import PROMPT_LAYOUT_ANALYSIS
        return await self._call_vision_model(
            prompt=PROMPT_LAYOUT_ANALYSIS,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_whole_page(
        self,
        image: np.ndarray,
        model_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        from oxr.model.prompts import PROMPT_WHOLE_PAGE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_WHOLE_PAGE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_chemical_structure(
        self,
        image: np.ndarray,
        model_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        from oxr.model.prompts import PROMPT_CHEMICAL_STRUCTURE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_CHEMICAL_STRUCTURE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )


ModelClient = OXRModelClient
model_client = OXRModelClient()
