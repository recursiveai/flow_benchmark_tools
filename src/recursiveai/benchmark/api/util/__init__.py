# Copyright 2024 Recursive AI

import json

from ..benchmark import Benchmark


def read_benchmarks_from_jsonl(jsonl_file: str) -> list[Benchmark]:
    with open(jsonl_file, "r") as f:
        jsons = [json.loads(line) for line in f]

    return [Benchmark(**bm) for bm in jsons]
