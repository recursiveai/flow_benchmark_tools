import logging
from functools import cached_property
from typing import AsyncIterable

from google.generativeai import GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from ._llm_model import ChatMessage, ChatResponseChunk, LLMModel

_logger = logging.getLogger(__name__)


class GoogleGemini(LLMModel):
    def __init__(self, name: str, context_window: int, output_window: int = -1):
        self._name = name
        self._context_window = context_window
        if output_window < 0:
            self._output_window = context_window
        else:
            self._output_window = output_window

        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def output_window(self) -> int:
        return self._output_window

    @cached_property
    def _client(self) -> GenerativeModel:
        return GenerativeModel(
            model_name=self._name, safety_settings=self._safety_settings
        )

    def count_tokens(self, message: str) -> int:
        return self._client.count_tokens(message).total_tokens

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

    async def async_chat_completion_stream(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> AsyncIterable[ChatResponseChunk] | None:
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
                stream=True,
            )
        except Exception:
            _logger.exception("Caught exception while running async_chat_completion")
            return None

        async def response_stream():
            async for content in response:
                yield ChatResponseChunk(delta=content.text, finish_reason=None)

        return response_stream()


GEMINI_1_5_FLASH = GoogleGemini(
    name="gemini-1.5-flash", context_window=1048576, output_window=8192
)
GEMINI_1_5_PRO = GoogleGemini(
    name="gemini-1.5-pro", context_window=1048576, output_window=8192
)
