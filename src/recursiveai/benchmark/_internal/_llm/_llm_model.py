# Copyright 2024 Recursive AI

from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    content: str
    role: Role


class LLMModel(ABC):
    def __init__(
        self, name: str, context_window: int, output_window: Optional[int] = None
    ):
        self._name = name
        self._context_window = context_window
        if output_window and output_window > 0:
            self._output_window = output_window
        else:
            self._output_window = context_window

    @property
    def name(self) -> str:
        return self._name

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def output_window(self) -> int:
        return self._output_window

    @abstractmethod
    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        raise NotImplementedError()
