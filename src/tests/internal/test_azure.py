# Copyright 2024 Recursive AI

import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._llm._azure_openai_gpt_model import (
    AZURE_GPT,
    AzureGPTX,
)
from recursiveai.benchmark._internal._llm._llm_model import ChatMessage


@pytest.fixture(autouse=True, scope="module")
def skip_if_api_key_not_defined():
    if os.getenv("AZURE_OPENAI_API_KEY") is None:
        pytest.skip(
            "Skipping Azure OpenAI test since AZURE_OPENAI_API_KEY is not defined"
        )
    if os.getenv("AZURE_OPENAI_DEPLOYMENT") is None:
        pytest.skip(
            "Skipping Azure OpenAI test since AZURE_OPENAI_DEPLOYMENT is not defined"
        )
    if os.getenv("AZURE_OPENAI_ENDPOINT") is None:
        pytest.skip(
            "Skipping Azure OpenAI test since AZURE_OPENAI_ENDPOINT is not defined"
        )


@pytest.fixture
def mock_model() -> AzureGPTX:
    model = AzureGPTX()
    model._client = Mock()
    return model


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[AZURE_GPT],
    ids=[AZURE_GPT.name],
)
async def test_async_chat_completion_success(model: AzureGPTX):
    chat = [
        ChatMessage(content="You are an AI assistant.", role="system"),
        ChatMessage(content="Very brief responses, ok?", role="user"),
        ChatMessage(content="OK", role="assistant"),
        ChatMessage(content="What's 2+2? Reply with a single number.", role="user"),
    ]
    response = await model.async_chat_completion(
        chat=chat, temperature=0.0, max_tokens=2
    )
    assert response == "4"


@pytest.mark.asyncio
async def test_async_chat_completion_failure(mock_model: AzureGPTX):
    mock_model._client.chat.completions.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion([])
    assert response == None
