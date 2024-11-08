# Copyright 2024 Recursive AI

from typing import Callable

import pytest

from recursiveai.benchmark.api import BenchmarkCase


@pytest.fixture
def sample_benchmark_case() -> BenchmarkCase:
    return BenchmarkCase(
        query="test_query",
        reference_answer="test_reference_answer",
        labels=["testing", "mocks"],
        extras={"extra_str": "test_string"},
    )


@pytest.fixture
def benchmark_case_list(sample_benchmark_case: BenchmarkCase) -> list[BenchmarkCase]:
    return [
        sample_benchmark_case.model_copy(deep=True),
        sample_benchmark_case.model_copy(deep=True),
        sample_benchmark_case.model_copy(deep=True),
    ]


@pytest.fixture
def callback_function() -> Callable[[str], str]:
    def f(query: str) -> str:
        return query

    return f
