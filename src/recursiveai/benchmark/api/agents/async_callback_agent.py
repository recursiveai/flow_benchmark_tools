import logging
from typing import Awaitable, Callable

from ..benchmark import Benchmark, BenchmarkResponse
from ..benchmark_agent import BenchmarkAgent
from ..exit_code import ExitCode

_logger = logging.getLogger(__name__)


class AsyncCallbackAgent(BenchmarkAgent):

    def __init__(self, async_callback: Callable[[str], Awaitable[str]]) -> None:
        super().__init__()
        self._async_callback = async_callback

    async def run_benchmark(self, benchmark: Benchmark) -> BenchmarkResponse:
        try:
            response = await self._async_callback(benchmark.query)
            exit_code = ExitCode.SUCCESS
        except Exception:
            _logger.exception(
                "Caught exception while running benchmark: %s", benchmark.query
            )
            response = None
            exit_code = ExitCode.FAILED
        return BenchmarkResponse(response=response, exit_code=exit_code)
