# Copyright 2024 Recursive AI

import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._llm._anthropic_claude_model import (
    CLAUDE_3_5_SONNET,
    CLAUDE_3_HAIKU,
    CLAUDE_3_OPUS,
    CLAUDE_3_SONNET,
    AnthropicClaude,
)
from recursiveai.benchmark._internal._llm._llm_model import ChatMessage


@pytest.fixture(autouse=True, scope="module")
def skip_if_api_key_not_defined() -> None:
    if os.getenv("ANTHROPIC_API_KEY") is None:
        pytest.skip("Skipping OpenAI test since ANTHROPIC_API_KEY is not defined")


@pytest.fixture
def mock_model() -> AnthropicClaude:
    model = AnthropicClaude(name="mock_model", context_window=8)
    model._client = Mock()
    return model


def test_no_output_window(mock_model: AnthropicClaude):
    assert mock_model.context_window == mock_model.output_window


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[CLAUDE_3_HAIKU, CLAUDE_3_SONNET, CLAUDE_3_OPUS, CLAUDE_3_5_SONNET],
    ids=[
        CLAUDE_3_HAIKU.name,
        CLAUDE_3_SONNET.name,
        CLAUDE_3_OPUS.name,
        CLAUDE_3_5_SONNET.name,
    ],
)
async def test_async_chat_completion_success(model: AnthropicClaude):
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
async def test_async_chat_completion_failure(mock_model: AnthropicClaude):
    mock_model._client.messages.create = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response == None
