import pytest

from recursiveai.benchmark.api.agents import CallbackAgent
from recursiveai.benchmark.api.exit_code import ExitCode


@pytest.fixture
def callback_function():
    def f(query: str) -> str:
        return query

    return f


@pytest.mark.asyncio
async def test_callback_agent_success(callback_function, dummy_benchmark):
    agent = CallbackAgent(callback=callback_function)
    response = await agent.run_benchmark(benchmark=dummy_benchmark)
    assert response.exit_code == ExitCode.SUCCESS
    assert response.response == callback_function(dummy_benchmark.query)


@pytest.mark.asyncio
async def test_callback_agent_failure(dummy_benchmark):
    def fail_func(s: str):
        raise Exception()

    agent = CallbackAgent(callback=fail_func)
    response = await agent.run_benchmark(benchmark=dummy_benchmark)
    assert response.exit_code == ExitCode.FAILED
    assert response.response == None
