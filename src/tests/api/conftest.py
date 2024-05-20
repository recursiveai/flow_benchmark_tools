import pytest

from recursiveai.benchmark.api import Benchmark, BenchmarkRun
from recursiveai.benchmark.api.agents import CallbackAgent


@pytest.fixture
def sample_benchmark():
    return Benchmark(
        query="test_query",
        reference_answer="test_reference_answer",
        labels=["testing", "mocks"],
        extras={"extra_str": "test_string"},
    )


@pytest.fixture
def benchmark_list(sample_benchmark):
    return [
        sample_benchmark.model_copy(deep=True),
        sample_benchmark.model_copy(deep=True),
        sample_benchmark.model_copy(deep=True),
    ]


@pytest.fixture
def callback_function():
    def f(query: str) -> str:
        return query

    return f


@pytest.fixture
def callback_agent(callback_function):
    return CallbackAgent(callback=callback_function)


@pytest.fixture
def benchmark_run(callback_agent, benchmark_list):
    return BenchmarkRun(agent=callback_agent, benchmarks=benchmark_list)
