# Copyright 2024 Recursive AI

from dataclasses import dataclass

from .benchmark import Benchmark
from .benchmark_agent import BenchmarkAgent


@dataclass
class BenchmarkRun:
    agent: BenchmarkAgent
    benchmark: Benchmark
