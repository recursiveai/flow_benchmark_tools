import pytest

from recursiveai.benchmark.api.benchmark import Benchmark


@pytest.fixture
def dummy_benchmark():
    return Benchmark(
        query="Is this a dummy benchmark?",
        reference_answer="Yes.",
        labels=["testing", "mocks"],
        extras={"extra_str": "Test string"},
    )
