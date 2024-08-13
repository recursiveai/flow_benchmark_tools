# Copyright 2024 Recursive AI

import logging
from typing import Optional

from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.generativeai import GenerativeModel
from google.generativeai.types import (
    AsyncGenerateContentResponse,
    ContentDict,
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)

from .._util import async_retry
from ._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)


class GoogleGemini(LLMModel):

    def _client(self, system_prompt: Optional[str] = None) -> GenerativeModel:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        return GenerativeModel(
            model_name=self._name,
            safety_settings=safety_settings,
            system_instruction=system_prompt,
        )

    async def async_chat_completion(
        self,
        chat: list[ChatMessage],
        **kwargs,
    ) -> str | None:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", self._output_window)
        timeout = kwargs.get("timeout", 60)

        (system, messages) = await self._convert_chat_to_messages(chat)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            candidate_count=1,
        )

        try:
            response: AsyncGenerateContentResponse = await self._completion(
                messages, system, config, timeout
            )
        except Exception:
            _logger.exception("Caught exception while running async_chat_completion")
            return None

        return response.text

    @async_retry(
        exc_tuple=(
            DeadlineExceeded,
            ResourceExhausted,
            InternalServerError,
            ServiceUnavailable,
        )
    )
    async def _completion(
        self,
        messages: list[ContentDict],
        system_prompt: str,
        config: GenerationConfig,
        timeout: float,
    ) -> AsyncGenerateContentResponse:
        return await self._client(system_prompt).generate_content_async(
            contents=messages,
            generation_config=config,
            request_options={"timeout": timeout},
            stream=False,
        )

    async def _convert_chat_to_messages(
        self, chat: list[ChatMessage]
    ) -> tuple[str, list[ContentDict]]:
        system = ""
        messages = []
        for msg in chat:
            match msg.role:
                case "system":
                    system = msg.content
                case "assistant":
                    messages.append(ContentDict(role="model", parts=[msg.content]))
                case "user":
                    messages.append(ContentDict(role="user", parts=[msg.content]))

        return (system, messages)


GEMINI_1_5_FLASH = GoogleGemini(
    name="gemini-1.5-flash", context_window=1048576, output_window=8192
)
GEMINI_1_5_PRO = GoogleGemini(
    name="gemini-1.5-pro", context_window=1048576, output_window=8192
)
