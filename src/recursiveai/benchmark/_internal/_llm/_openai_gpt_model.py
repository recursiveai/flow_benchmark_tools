import logging
from functools import cached_property
from typing import AsyncIterable

from openai import AsyncOpenAI
from tiktoken import encoding_for_model

from ._llm_model import ChatMessage, ChatResponseChunk, LLMModel

_logger = logging.getLogger(__name__)


class GPTX(LLMModel):
    def __init__(self, name: str, context_window: int, output_window: int = -1):
        self._name = name
        self._context_window = context_window
        if output_window < 0:
            self._output_window = context_window
        else:
            self._output_window = output_window

        try:
            self._encoding = encoding_for_model(name)
        except KeyError:
            _logger.warning(f"Could not get encoding for model {name}")
            self._encoding = None

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
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    def count_tokens(self, message: str) -> int:
        if self._encoding is None:
            _logger.error("encoding is None")
            return 0
        return len(self._encoding.encode(message))

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.3)
        max_tokens = kwargs.get("max_tokens", 256)
        timeout = kwargs.get("timeout", 30)

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
            _logger.error(
                "Caught exception when running async_chat_completion", exc_info=1
            )
            return None

        return response.choices[0].message.content

    async def async_chat_completion_stream(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> AsyncIterable[ChatResponseChunk] | None:
        temperature = kwargs.get("temperature", 0.3)
        max_tokens = kwargs.get("max_tokens", 1024)
        timeout = kwargs.get("timeout", 30)

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
            _logger.error(
                "Caught exception when running async_chat_completion_stream", exc_info=1
            )
            return None

        async def response_stream():
            async for resp in response:
                choice = resp.choices[0]
                yield ChatResponseChunk(
                    delta=choice.delta.content,
                    finish_reason=choice.finish_reason,
                )

        return response_stream()


GPT_3_5_TURBO = GPTX(name="gpt-3.5-turbo", context_window=16385, output_window=4096)
GPT_4_TURBO_PREVIEW = GPTX(
    name="gpt-4-turbo-preview", context_window=128000, output_window=4096
)
GPT_4_O = GPTX(name="gpt-4o", context_window=128000, output_window=4096)
