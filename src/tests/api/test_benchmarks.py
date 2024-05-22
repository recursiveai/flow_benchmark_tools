import errno
import json
import os
from unittest.mock import Mock

import pytest

from recursiveai.benchmark.api import Benchmark, BenchmarkRun
from recursiveai.benchmark.api.util import (
    create_run_from_benchmark_jsonl,
    read_benchmarks_from_jsonl,
)

_TEST_REFERENCE_ANSWER_FILE = "test_reference_answer.txt"
_TEST_REFERENCE_ANSWER = "This is a reference answer"
_TEST_BENCHMARK_JSONL_FILE = "benchmarks.jsonl"


@pytest.fixture
def reference_answer_file():
    with open(_TEST_REFERENCE_ANSWER_FILE, "w") as f:
        f.write(_TEST_REFERENCE_ANSWER)

    yield

    try:
        os.remove(_TEST_REFERENCE_ANSWER_FILE)
    except OSError as err:
        if err.errno == errno.ENOENT:
            pass
        else:
            raise


@pytest.fixture
def benchmark_jsons():
    return [
        {
            "query": "First query",
            "reference_answer": "First reference answer",
            "labels": ["testing", "mocks"],
        },
        {
            "query": "Second query",
            "reference_answer": "Second reference answer",
            "labels": ["testing", "mocks"],
            "extras": {"extra_str": "Test string"},
        },
    ]


@pytest.fixture
def benchmark_jsonl(benchmark_jsons):
    with open(_TEST_BENCHMARK_JSONL_FILE, "w") as f:
        for entry in benchmark_jsons:
            json.dump(entry, f)
            f.write("\n")

    yield

    try:
        os.remove(_TEST_BENCHMARK_JSONL_FILE)
    except OSError as err:
        if err.errno == errno.ENOENT:
            pass
        else:
            raise


def test_error_on_no_reference_answer():
    with pytest.raises(ValueError):
        Benchmark(query="This is a test query")


def test_reference_answer_from_file(reference_answer_file):
    benchmark = Benchmark(
        query="This is a test query", reference_answer_file=_TEST_REFERENCE_ANSWER_FILE
    )
    assert benchmark.reference_answer == _TEST_REFERENCE_ANSWER


def test_read_jsonl_from_file(benchmark_jsons, benchmark_jsonl):
    benchmarks = read_benchmarks_from_jsonl(_TEST_BENCHMARK_JSONL_FILE)

    assert len(benchmarks) == len(benchmark_jsons)

    for idx, benchmark in enumerate(benchmarks):
        assert (
            benchmark.model_dump(exclude_none=True, exclude_unset=True)
            == benchmark_jsons[idx]
        )


def test_create_run_from_benchmark_jsonl(benchmark_jsons, benchmark_jsonl):
    run = create_run_from_benchmark_jsonl(
        agent=Mock(), jsonl_file=_TEST_BENCHMARK_JSONL_FILE
    )

    assert isinstance(run, BenchmarkRun)
    assert len(run.benchmarks) == len(benchmark_jsons)
    for idx, benchmark in enumerate(run.benchmarks):
        assert (
            benchmark.model_dump(exclude_none=True, exclude_unset=True)
            == benchmark_jsons[idx]
        )
