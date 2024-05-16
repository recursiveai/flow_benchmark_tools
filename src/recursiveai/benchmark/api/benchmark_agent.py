from abc import ABC, abstractmethod

from .benchmark import Benchmark, BenchmarkResponse


class BenchmarkAgent(ABC):
    """
    Interface used to define the semantic agents.

    Agents implementing this interface are able to run semantic benchmarks, i.e.,
    benchmarks that take a text query as input and produce a text response as output.
    """

    @abstractmethod
    async def run_benchmark(self, benchmark: Benchmark) -> BenchmarkResponse:
        raise NotImplementedError()
