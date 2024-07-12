# Copyright 2024 Recursive AI

from ..benchmark import Benchmark
from ..benchmark_agent import BenchmarkAgent
from ..benchmark_case import BenchmarkCase
from ..benchmark_run import BenchmarkRun


def read_benchmark_from_jsonl(jsonl_file: str) -> Benchmark:
    with open(jsonl_file, "r") as f:
        cases = [BenchmarkCase.model_validate_json(line) for line in f]

    return Benchmark(cases=cases)


def create_run_from_jsonl(agent: BenchmarkAgent, jsonl_file: str) -> BenchmarkRun:
    benchmark = read_benchmark_from_jsonl(jsonl_file=jsonl_file)
    return BenchmarkRun(agent=agent, benchmark=benchmark)
