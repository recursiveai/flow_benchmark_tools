import pytest

from recursiveai.benchmark.api import ExitCode
from recursiveai.benchmark.api.agents import CallbackAgent


@pytest.mark.asyncio
async def test_callback_agent_success(callback_function, sample_benchmark):
    agent = CallbackAgent(callback=callback_function)
    response = await agent.run_benchmark(benchmark=sample_benchmark)
    assert response.exit_code == ExitCode.SUCCESS
    assert response.response == callback_function(sample_benchmark.query)


@pytest.mark.asyncio
async def test_callback_agent_failure(sample_benchmark):
    def fail_func(_: str):
        raise Exception()

    agent = CallbackAgent(callback=fail_func)
    response = await agent.run_benchmark(benchmark=sample_benchmark)
    assert response.exit_code == ExitCode.FAILED
    assert response.response == None
