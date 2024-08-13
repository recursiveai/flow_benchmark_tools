# Copyright 2024 Recursive AI
import os
from functools import cached_property

from openai import AsyncAzureOpenAI

from ._llm_model import ChatMessage
from ._openai_gpt_model import GPTX

_DEFAULT_OPENAI_API_VERSION = "2024-06-01"


class AzureGPTX(GPTX):
    def __init__(self):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            deployment = ""

        self._api_version = os.getenv("OPENAI_API_VERSION")
        if not self._api_version:
            self._api_version = _DEFAULT_OPENAI_API_VERSION

        super().__init__(name=deployment, context_window=0)

    @cached_property
    def _client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_deployment=self.name, api_version=self._api_version
        )

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        kwargs["max_tokens"] = None
        return await super().async_chat_completion(chat, **kwargs)


AZURE_GPT = AzureGPTX()
