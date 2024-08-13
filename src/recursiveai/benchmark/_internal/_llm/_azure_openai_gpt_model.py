# Copyright 2024 Recursive AI
import os
import logging

from functools import cached_property

from openai import AsyncAzureOpenAI
from ._openai_gpt_model import GPTX

from ._llm_model import ChatMessage

_logger = logging.getLogger(__name__)

class AzureGPTX(GPTX):
    def __init__(self, context_window: int = 0):
        super().__init__(
            os.environ["AZURE_OPENAI_DEPLOYMENT"], 
            context_window
        )
    
    @cached_property
    def _client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_deployment=self.name,
        )

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        
        kwargs["max_tokens"] = None
        return await super().async_chat_completion(chat, **kwargs)
    
AZURE_GPT = AzureGPTX()