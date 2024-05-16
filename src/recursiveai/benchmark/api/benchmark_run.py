from .benchmark import Benchmark
from .benchmark_agent import BenchmarkAgent


class BenchmarkRun:

    def __init__(self, agent: BenchmarkAgent, benchmarks: list[Benchmark]) -> None:
        self.agent = agent
        self.benchmarks = benchmarks
