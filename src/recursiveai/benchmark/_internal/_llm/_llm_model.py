# Copyright 2024 Recursive AI

from abc import ABC, abstractmethod
from typing import AsyncIterable, Literal

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    content: str
    role: Literal["user", "assistant", "system"]


class ChatResponseChunk(BaseModel):
    delta: str | None = None
    finish_reason: str | None = None

    model_config = ConfigDict()


class LLMModel(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def context_window(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_window(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def count_tokens(self, message: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        raise NotImplementedError()

    @abstractmethod
    async def async_chat_completion_stream(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> AsyncIterable[ChatResponseChunk] | None:
        raise NotImplementedError()
