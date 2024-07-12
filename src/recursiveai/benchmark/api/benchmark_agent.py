# Copyright 2024 Recursive AI

from abc import ABC, abstractmethod

from .benchmark import Benchmark
from .benchmark_case import BenchmarkCase, BenchmarkCaseResponse


class BenchmarkAgent(ABC):
    """
    Interface used to define the semantic agents.

    Agents implementing this interface are able to run semantic benchmarks, i.e.,
    benchmarks that take a text query as input and produce a text response as output.
    """

    @abstractmethod
    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        raise NotImplementedError()

    async def before_run(self, benchmark: Benchmark) -> None:
        pass

    async def before_case(self, case: BenchmarkCase) -> None:
        pass

    async def after_case(self, case: BenchmarkCase) -> None:
        pass

    async def after_run(self, benchmark: Benchmark) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
