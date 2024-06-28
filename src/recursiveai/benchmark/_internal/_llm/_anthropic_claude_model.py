import logging
from functools import cached_property
from typing import AsyncIterable

from anthropic import AsyncAnthropic

from ._llm_model import ChatMessage, ChatResponseChunk, LLMModel

_logger = logging.getLogger(__name__)


class AnthropicClaude(LLMModel):
    def __init__(self, name: str, context_window: int, output_window: int = -1):
        self._name = name
        self._context_window = context_window
        if output_window < 0:
            self._output_window = context_window
        else:
            self._output_window = output_window

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
    def _client(self) -> AsyncAnthropic:
        return AsyncAnthropic()

    def count_tokens(self, message: str) -> int:
        _logger.warning("Tokenizer not available for Anthropic Claude models")
        return 0

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        timeout = kwargs.get("timeout", 60)

        messages = []
        system_prompt = ""
        for msg in chat:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        try:
            response = await self._client.messages.create(
                model=self._name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                system=system_prompt,
                messages=messages,
                stream=False,
            )
        except Exception:
            _logger.exception("Caught exception while running async_chat_completion")
            return None

        return response.content[0].text

    async def async_chat_completion_stream(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> AsyncIterable[ChatResponseChunk] | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)
        timeout = kwargs.get("timeout", 60)

        messages = []
        system_prompt = ""
        for msg in chat:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        try:
            response = await self._client.messages.create(
                model=self._name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                system=system_prompt,
                messages=messages,
                stream=True,
            )
        except Exception:
            _logger.exception("Caught exception while running async_chat_completion")
            return None

        async def response_stream():
            async for content in response:
                if content.type == "content_block_delta":
                    yield ChatResponseChunk(
                        delta=content.delta.text, finish_reason=None
                    )

        return response_stream()


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
