# Copyright 2024 Recursive AI

import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._llm._google_gemini_model import (
    GEMINI_1_5_FLASH,
    GEMINI_1_5_PRO,
    GoogleGemini,
)
from recursiveai.benchmark._internal._llm._llm_model import ChatMessage


@pytest.fixture(autouse=True, scope="module")
def skip_if_api_key_not_defined() -> None:
    if os.getenv("GOOGLE_API_KEY") is None:
        pytest.skip("Skipping OpenAI test since GOOGLE_API_KEY is not defined")


@pytest.fixture
def mock_model() -> GoogleGemini:
    model = GoogleGemini(name="mock_model", context_window=8)
    model._client = Mock()
    return model


def test_gemini_1_5_flash_context_window():
    assert GEMINI_1_5_FLASH.context_window == 1048576


def test_gemini_1_5_pro_context_window():
    assert GEMINI_1_5_PRO.context_window == 1048576


def test_no_output_window(mock_model: GoogleGemini):
    assert mock_model.context_window == mock_model.output_window


@pytest.mark.asyncio(scope="module")
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[GEMINI_1_5_FLASH, GEMINI_1_5_PRO],
    ids=[GEMINI_1_5_FLASH.name, GEMINI_1_5_PRO.name],
)
async def test_async_chat_completion_success(model: GoogleGemini):
    chat = [
        ChatMessage(content="You are an AI assistant.", role="system"),
        ChatMessage(content="Very brief responses, ok?", role="user"),
        ChatMessage(content="OK", role="assistant"),
        ChatMessage(content="What's 2+2? Reply with a single number.", role="user"),
    ]
    response = await model.async_chat_completion(
        chat=chat, temperature=0.0, max_tokens=2
    )
    assert response
    assert response.strip() == "4"


@pytest.mark.asyncio
async def test_async_chat_completion_failure(mock_model: GoogleGemini):
    mock_model._client.generate_content_async = AsyncMock(side_effect=Exception())
    response = await mock_model.async_chat_completion(
        chat=[], temperature=0.0, max_tokens=2
    )
    assert response is None
