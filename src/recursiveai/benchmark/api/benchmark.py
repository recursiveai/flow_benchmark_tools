# Copyright 2024 Recursive AI

from typing import Any

from pydantic import BaseModel

from .benchmark_case import BenchmarkCase


class Benchmark(BaseModel):
    cases: list[BenchmarkCase]
    extras: dict[str, Any] | None = None
