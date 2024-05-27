import logging
from typing import Callable

from ..benchmark import Benchmark, BenchmarkResponse
from ..benchmark_agent import BenchmarkAgent
from ..exit_code import ExitCode

_logger = logging.getLogger(__name__)


class CallbackAgent(BenchmarkAgent):

    def __init__(self, callback: Callable[[str], str]) -> None:
        super().__init__()
        self._callback = callback

    async def run_benchmark(self, benchmark: Benchmark) -> BenchmarkResponse:
        try:
            response = self._callback(benchmark.query)
            exit_code = ExitCode.SUCCESS
        except Exception:
            _logger.exception(
                "Caught exception while running benchmark: %s", benchmark.query
            )
            response = None
            exit_code = ExitCode.FAILED
        return BenchmarkResponse(response=response, exit_code=exit_code)
