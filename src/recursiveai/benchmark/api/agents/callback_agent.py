# Copyright 2024 Recursive AI

import logging
from typing import Callable

from ..benchmark_agent import BenchmarkAgent
from ..benchmark_case import BenchmarkCase, BenchmarkCaseResponse
from ..exit_code import ExitCode

_logger = logging.getLogger(__name__)


class CallbackAgent(BenchmarkAgent):

    def __init__(self, callback: Callable[[str], str]) -> None:
        super().__init__()
        self._callback = callback

    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        try:
            response = self._callback(case.query)
            exit_code = ExitCode.SUCCESS
        except Exception:
            _logger.exception(
                "Caught exception while running benchmark: %s", case.query
            )
            response = None
            exit_code = ExitCode.FAILED
        return BenchmarkCaseResponse(response=response, exit_code=exit_code)
