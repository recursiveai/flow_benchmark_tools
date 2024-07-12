# Copyright 2024 Recursive AI

import pytest

from recursiveai.benchmark.api import BenchmarkCase


@pytest.fixture
def sample_benchmark_case():
    return BenchmarkCase(
        query="test_query",
        reference_answer="test_reference_answer",
        labels=["testing", "mocks"],
        extras={"extra_str": "test_string"},
    )


@pytest.fixture
def benchmark_case_list(sample_benchmark_case):
    return [
        sample_benchmark_case.model_copy(deep=True),
        sample_benchmark_case.model_copy(deep=True),
        sample_benchmark_case.model_copy(deep=True),
    ]


@pytest.fixture
def callback_function():
    def f(query: str) -> str:
        return query

    return f
