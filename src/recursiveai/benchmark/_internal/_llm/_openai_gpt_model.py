# Copyright 2024 Recursive AI

import logging
from functools import cached_property

from openai import APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)

from .._util import async_retry
from ._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)


class GPTX(LLMModel):

    @cached_property
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", self._output_window)
        timeout = kwargs.get("timeout", 60)

        messages = await self._convert_chat_to_messages(chat)
        request = CompletionCreateParamsNonStreaming(
            model=self._name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stream=False,
        )

        try:
            response: ChatCompletion = await self._completion(request, timeout)
        except Exception:
            _logger.exception("Caught exception when running async_chat_completion")
            return None

        return response.choices[0].message.content

    @async_retry(exc_tuple=(APITimeoutError, RateLimitError, InternalServerError))
    async def _completion(
        self, request: CompletionCreateParamsNonStreaming, timeout: float
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            **request,
            timeout=timeout,
        )

    async def _convert_chat_to_messages(
        self, chat: list[ChatMessage]
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in chat
        ]

        return messages


GPT_3_5_TURBO = GPTX(name="gpt-3.5-turbo", context_window=16385, output_window=4096)
GPT_4_TURBO_PREVIEW = GPTX(
    name="gpt-4-turbo-preview", context_window=128000, output_window=4096
)
GPT_4_O = GPTX(name="gpt-4o", context_window=128000, output_window=4096)
