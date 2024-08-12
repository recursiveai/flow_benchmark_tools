# Copyright 2024 Recursive AI

import logging
from functools import cached_property

from openai import AsyncOpenAI

from ._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)


class GPTX(LLMModel):
    def __init__(self, name: str, context_window: int, output_window: int = -1):
        super().__init__(name, context_window, output_window)

    @cached_property
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        timeout = kwargs.get("timeout", 60)

        messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in chat
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stream=False,
                timeout=timeout,
            )
        except Exception:
            _logger.exception("Caught exception when running async_chat_completion")
            return None

        return response.choices[0].message.content


GPT_3_5_TURBO = GPTX(name="gpt-3.5-turbo", context_window=16385, output_window=4096)
GPT_4_TURBO_PREVIEW = GPTX(
    name="gpt-4-turbo-preview", context_window=128000, output_window=4096
)
GPT_4_O = GPTX(name="gpt-4o", context_window=128000, output_window=4096)
