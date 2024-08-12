# Copyright 2024 Recursive AI

import logging
from functools import cached_property

from google.generativeai import GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from ._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)


class GoogleGemini(LLMModel):
    def __init__(self, name: str, context_window: int, output_window: int = -1):
        super().__init__(name, context_window, output_window)

        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    @cached_property
    def _client(self) -> GenerativeModel:
        return GenerativeModel(
            model_name=self._name, safety_settings=self._safety_settings
        )

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        timeout = kwargs.get("timeout", 60)

        messages = []
        for msg in chat:
            match (msg.role):
                case "system":
                    messages.append({"role": "user", "parts": [msg.content]})
                    messages.append({"role": "model", "parts": ["OK"]})
                case "user":
                    messages.append({"role": "user", "parts": [msg.content]})
                case "assistant":
                    messages.append({"role": "model", "parts": [msg.content]})

        try:
            response = await self._client.generate_content_async(
                contents=messages,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                request_options={"timeout": timeout},
                stream=False,
            )
        except Exception:
            _logger.exception("Caught exception while running async_chat_completion")
            return None

        return response.text


GEMINI_1_5_FLASH = GoogleGemini(
    name="gemini-1.5-flash", context_window=1048576, output_window=8192
)
GEMINI_1_5_PRO = GoogleGemini(
    name="gemini-1.5-pro", context_window=1048576, output_window=8192
)
