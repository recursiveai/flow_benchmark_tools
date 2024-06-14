import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._llm._anthropic_claude_model import (
    CLAUDE_3_HAIKU,
    CLAUDE_3_OPUS,
    CLAUDE_3_SONNET,
    AnthropicClaude,
)
from recursiveai.benchmark._internal._llm._llm_model import ChatMessage


@pytest.fixture(autouse=True, scope="module")
def skip_if_api_key_not_defined():
    if os.getenv("ANTHROPIC_API_KEY") is None:
        pytest.skip("Skipping OpenAI test since ANTHROPIC_API_KEY is not defined")


@pytest.fixture
def mock_model():
    model = AnthropicClaude(name="mock_model", context_window=8)
    model._client = Mock()
    return model


def test_no_output_window(mock_model):
    assert mock_model.context_window == mock_model.output_window


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[CLAUDE_3_HAIKU, CLAUDE_3_SONNET, CLAUDE_3_OPUS],
    ids=[CLAUDE_3_HAIKU.name, CLAUDE_3_SONNET.name, CLAUDE_3_OPUS.name],
)
async def test_async_chat_completion_success(model):
    chat = [
        ChatMessage(content="You are an AI assistant.", role="system"),
        ChatMessage(content="What's 2+2? Reply with a single number.", role="user"),
    ]
    response = await model.async_chat_completion(
        chat=chat, temperature=0.0, max_tokens=2
    )
    assert response == "4"


@pytest.mark.asyncio
async def test_async_chat_completion_failure(mock_model):
    mock_model._client.messages.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response == None


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[CLAUDE_3_HAIKU, CLAUDE_3_SONNET, CLAUDE_3_OPUS],
    ids=[CLAUDE_3_HAIKU.name, CLAUDE_3_SONNET.name, CLAUDE_3_OPUS.name],
)
async def test_async_chat_completion_stream_success(model):
    chat = [
        ChatMessage(content="You are an AI assistant.", role="system"),
        ChatMessage(content="What's 2+2? Reply with a single number.", role="user"),
    ]
    response_stream = await model.async_chat_completion_stream(
        chat=chat, temperature=0.0, max_tokens=256
    )
    response = ""
    async for chunk in response_stream:
        if chunk and chunk.delta:
            response += chunk.delta

    assert response == "4"


@pytest.mark.asyncio
async def test_async_chat_completion_stream_failure(mock_model):
    mock_model._client.messages.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion_stream(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response == None


def test_count_tokens(mock_model):
    num_tokens = mock_model.count_tokens("Hi")
    assert num_tokens == 0
