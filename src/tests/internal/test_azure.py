# Copyright 2024 Recursive AI

import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._llm._llm_model import ChatMessage
from recursiveai.benchmark._internal._llm._azure_openai_gpt_model import (
    AZURE_GPT_3_5_TURBO,
    AZURE_GPT_4_O,
    AZURE_GPT_4_TURBO_PREVIEW,
    AzureGPTX,
)


@pytest.fixture(autouse=True, scope="module")
def skip_if_api_key_not_defined():
    if os.getenv("OPENAI_API_KEY") is None:
        pytest.skip("Skipping Azure OpenAI test since OPENAI_API_KEY is not defined")
    if os.getenv("AZURE_OPENAI_DEPLOYMENT") is None:
        pytest.skip("Skipping Azure OpenAI test since AZURE_OPENAI_DEPLOYMENT is not defined")


@pytest.fixture
def mock_model():
    model = AzureGPTX(name="mock_model", context_window=8)
    model._client = Mock()
    return model


def test_gpt_3_5_turbo_context_window():
    assert AZURE_GPT_3_5_TURBO.context_window == 16385


def test_gpt_4_turbo_context_window():
    assert AZURE_GPT_4_TURBO_PREVIEW.context_window == 128000


def test_gpt_4o_context_window():
    assert AZURE_GPT_4_O.context_window == 128000


def test_no_output_window(mock_model):
    assert mock_model.context_window == mock_model.output_window


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[AZURE_GPT_3_5_TURBO, AZURE_GPT_4_TURBO_PREVIEW, AZURE_GPT_4_O],
    ids=[AZURE_GPT_3_5_TURBO.name, AZURE_GPT_4_TURBO_PREVIEW.name, AZURE_GPT_4_O.name],
)
async def test_async_chat_completion_success(model):
    chat = [ChatMessage(content="What's 2+2? Reply with a single number.", role="user")]
    response = await model.async_chat_completion(
        chat=chat, temperature=0.0, max_tokens=2
    )
    assert response == "4"


@pytest.mark.asyncio
async def test_async_chat_completion_failure(mock_model):
    mock_model._client.chat.completions.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response == None


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[AZURE_GPT_3_5_TURBO, AZURE_GPT_4_TURBO_PREVIEW, AZURE_GPT_4_O],
    ids=[AZURE_GPT_3_5_TURBO.name, AZURE_GPT_4_TURBO_PREVIEW.name, AZURE_GPT_4_O.name],
)
async def test_async_chat_completion_stream_success(model):
    chat = [ChatMessage(content="What's 2+2? Reply with a single number.", role="user")]
    response_stream = await model.async_chat_completion_stream(
        chat=chat, temperature=0.0, max_tokens=2
    )
    response = ""
    async for chunk in response_stream:
        if chunk and chunk.delta:
            response += chunk.delta

    assert response == "4"


@pytest.mark.asyncio
async def test_async_chat_completion_stream_failure(mock_model):
    mock_model._client.chat.completions.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion_stream(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response == None


@pytest.mark.parametrize(
    argnames="model",
    argvalues=[AZURE_GPT_3_5_TURBO, AZURE_GPT_4_TURBO_PREVIEW, AZURE_GPT_4_O],
    ids=[AZURE_GPT_3_5_TURBO.name, AZURE_GPT_4_TURBO_PREVIEW.name, AZURE_GPT_4_O.name],
)
def test_count_tokens_success(model):
    num_tokens = model.count_tokens("Hi")
    assert num_tokens == 1


def test_count_tokens_failure(mock_model):
    num_tokens = mock_model.count_tokens("Hi")
    assert num_tokens == 0
