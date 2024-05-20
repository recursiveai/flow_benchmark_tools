from dataclasses import dataclass

from .benchmark import Benchmark
from .benchmark_agent import BenchmarkAgent


@dataclass
class BenchmarkRun:
    agent: BenchmarkAgent
    benchmarks: list[Benchmark]
