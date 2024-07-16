# Copyright 2024 Recursive AI

import logging
from typing import Awaitable, Callable

from ..benchmark_agent import BenchmarkAgent
from ..benchmark_case import BenchmarkCase, BenchmarkCaseResponse
from ..exit_code import ExitCode

_logger = logging.getLogger(__name__)


class AsyncCallbackAgent(BenchmarkAgent):

    def __init__(self, async_callback: Callable[[str], Awaitable[str]]) -> None:
        super().__init__()
        self._async_callback = async_callback

    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        try:
            response = await self._async_callback(case.query)
            exit_code = ExitCode.SUCCESS
        except Exception:
            _logger.exception(
                "Caught exception while running benchmark: %s", case.query
            )
            response = None
            exit_code = ExitCode.FAILED
        return BenchmarkCaseResponse(response=response, exit_code=exit_code)
