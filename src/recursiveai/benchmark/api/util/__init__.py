# Copyright 2024 Recursive AI
from ..benchmark import Benchmark


def read_benchmarks_from_jsonl(jsonl_file: str) -> list[Benchmark]:
    with open(jsonl_file, "r") as f:
        benchmarks = [Benchmark.model_validate_json(line) for line in f]

    return benchmarks
