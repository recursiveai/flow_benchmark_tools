# Copyright 2024 Recursive AI
from ..benchmark import Benchmark
from ..benchmark_agent import BenchmarkAgent
from ..benchmark_run import BenchmarkRun


def read_benchmarks_from_jsonl(jsonl_file: str) -> list[Benchmark]:
    with open(jsonl_file, "r") as f:
        benchmarks = [Benchmark.model_validate_json(line) for line in f]

    return benchmarks


def create_run_from_benchmark_jsonl(
    agent: BenchmarkAgent, jsonl_file: str
) -> BenchmarkRun:
    benchmarks = read_benchmarks_from_jsonl(jsonl_file=jsonl_file)
    return BenchmarkRun(agent=agent, benchmarks=benchmarks)
