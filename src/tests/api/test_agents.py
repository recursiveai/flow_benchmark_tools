# Copyright 2024 Recursive AI

from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark.api import ExitCode
from recursiveai.benchmark.api.agents import AsyncCallbackAgent, CallbackAgent


@pytest.mark.asyncio
async def test_callback_agent_success(sample_benchmark_case):
    callback = Mock(return_value="test")
    agent = CallbackAgent(callback=callback)
    response = await agent.run_benchmark_case(case=sample_benchmark_case)
    assert response.exit_code == ExitCode.SUCCESS
    assert response.response == "test"


@pytest.mark.asyncio
async def test_callback_agent_failure(sample_benchmark_case):
    def fail_func(_: str):
        raise Exception()

    agent = CallbackAgent(callback=fail_func)
    response = await agent.run_benchmark_case(case=sample_benchmark_case)
    assert response.exit_code == ExitCode.FAILED
    assert response.response == None


@pytest.mark.asyncio
async def test_async_callback_agent_success(sample_benchmark_case):
    async_callback = AsyncMock(return_value="test")
    agent = AsyncCallbackAgent(async_callback=async_callback)
    response = await agent.run_benchmark_case(case=sample_benchmark_case)
    assert response.exit_code == ExitCode.SUCCESS
    assert response.response == "test"


@pytest.mark.asyncio
async def test_async_callback_agent_failure(sample_benchmark_case):
    async def fail_func(_: str):
        raise Exception()

    agent = AsyncCallbackAgent(async_callback=fail_func)
    response = await agent.run_benchmark_case(case=sample_benchmark_case)
    assert response.exit_code == ExitCode.FAILED
    assert response.response == None
