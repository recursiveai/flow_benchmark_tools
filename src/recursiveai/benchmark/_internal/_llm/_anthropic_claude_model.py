# Copyright 2024 Recursive AI

import logging
from functools import cached_property

from anthropic import (
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)
from anthropic.types import Message
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from .._util import async_retry
from ._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)


class AnthropicClaude(LLMModel):

    @cached_property
    def _client(self) -> AsyncAnthropic:
        return AsyncAnthropic()

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", self._output_window)
        timeout = kwargs.get("timeout", 60)

        (system, messages) = await self._convert_chat_to_messages(chat)
        request = MessageCreateParamsNonStreaming(
            model=self.name,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        try:
            response: Message = await self._completion(request, timeout)
        except Exception:
            _logger.exception("Caught exception while running async_chat_completion")
            return None

        return response.content[0].text

    @async_retry(exc_tuple=(APITimeoutError, RateLimitError, InternalServerError))
    async def _completion(
        self, request: MessageCreateParamsNonStreaming, timeout: float
    ) -> Message:
        return await self._client.messages.create(**request, timeout=timeout)

    async def _convert_chat_to_messages(
        self, chat: list[ChatMessage]
    ) -> tuple[str, list[dict[str, str]]]:
        system = ""
        messages = []
        for msg in chat:
            if msg.role == "system":
                system = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})
        return (system, messages)


CLAUDE_3_OPUS = AnthropicClaude(
    name="claude-3-opus-20240229", context_window=200000, output_window=4096
)
CLAUDE_3_SONNET = AnthropicClaude(
    name="claude-3-sonnet-20240229", context_window=200000, output_window=4096
)
CLAUDE_3_5_SONNET = AnthropicClaude(
    name="claude-3-5-sonnet-20240620", context_window=200000, output_window=4096
)
CLAUDE_3_HAIKU = AnthropicClaude(
    name="claude-3-haiku-20240307", context_window=200000, output_window=4096
)
