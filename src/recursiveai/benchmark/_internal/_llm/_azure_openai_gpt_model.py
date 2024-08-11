# Copyright 2024 Recursive AI
import os
import logging

from functools import cached_property
from typing import AsyncIterable

from openai import AsyncAzureOpenAI
from ._openai_gpt_model import GPTX

from ._llm_model import ChatMessage, ChatResponseChunk

_logger = logging.getLogger(__name__)

class AzureGPTX(GPTX):
    def __init__(self, name: str, context_window: int, output_window: int = -1):
        super().__init__(name, context_window, output_window=output_window)
    
    @cached_property
    def _client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )

    
    async def async_chat_completion_stream(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> AsyncIterable[ChatResponseChunk] | None:
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
                stream=True,
                timeout=timeout,
            )
        except Exception:
            _logger.exception(
                "Caught exception when running async_chat_completion_stream"
            )
            return None
        async def response_stream():
            async for resp in response:
                if len(resp.choices) > 0:
                    choice = resp.choices[0]
                    yield ChatResponseChunk(
                        delta=choice.delta.content,
                        finish_reason=choice.finish_reason,
                    )

        return response_stream()

AZURE_GPT_3_5_TURBO = AzureGPTX(name="gpt-3.5-turbo", context_window=16385, output_window=4096)
AZURE_GPT_4_TURBO_PREVIEW = AzureGPTX(
    name="gpt-4-turbo-preview", context_window=128000, output_window=4096
)
AZURE_GPT_4_O = AzureGPTX(name="gpt-4o", context_window=128000, output_window=4096)
