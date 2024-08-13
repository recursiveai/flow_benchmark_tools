# Copyright 2024 Recursive AI

import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._llm._llm_model import ChatMessage
from recursiveai.benchmark._internal._llm._openai_gpt_model import (
    GPT_3_5_TURBO,
    GPT_4_O,
    GPT_4_TURBO_PREVIEW,
    GPTX,
)


@pytest.fixture(autouse=True, scope="module")
def skip_if_api_key_not_defined() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        pytest.skip("Skipping OpenAI test since OPENAI_API_KEY is not defined")


@pytest.fixture
def mock_model() -> GPTX:
    model = GPTX(name="mock_model", context_window=8)
    model._client = Mock()
    return model


def test_gpt_3_5_turbo_context_window():
    assert GPT_3_5_TURBO.context_window == 16385


def test_gpt_4_turbo_context_window():
    assert GPT_4_TURBO_PREVIEW.context_window == 128000


def test_gpt_4o_context_window():
    assert GPT_4_O.context_window == 128000


def test_no_output_window(mock_model: GPTX):
    assert mock_model.context_window == mock_model.output_window


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[GPT_3_5_TURBO, GPT_4_TURBO_PREVIEW, GPT_4_O],
    ids=[GPT_3_5_TURBO.name, GPT_4_TURBO_PREVIEW.name, GPT_4_O.name],
)
async def test_async_chat_completion_success(model: GPTX):
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
async def test_async_chat_completion_failure(mock_model: GPTX):
    mock_model._client.chat.completions.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response == None
