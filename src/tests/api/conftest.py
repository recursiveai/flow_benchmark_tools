import pytest

from recursiveai.benchmark.api import Benchmark


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
