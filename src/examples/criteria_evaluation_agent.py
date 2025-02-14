# Copyright 2024 Recursive AI

import asyncio
import logging

from recursiveai.benchmark.api import (
    BenchmarkAgent,
    BenchmarkCase,
    BenchmarkCaseResponse,
    Evaluator,
    ExitCode,
)
from recursiveai.benchmark.api.benchmark_runner import CriteriaBenchmarkRunner
from recursiveai.benchmark.api.util import create_run_from_jsonl

_logger = logging.getLogger(__name__)


class CriteriaAgent(BenchmarkAgent):
    def __init__(
        self,
    ) -> None:
        pass

    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        """Because we are evaluating the quality of already generated answers,
        the BenchmarkCaseResponse will simply repackage the query in BenchmarkCase"""

        if "criteria" in case.extras:
            return BenchmarkCaseResponse(
                response=case.query, extras=case.extras, exit_code=ExitCode.SUCCESS
            )
        else:
            _logger.error("No criteria was specified to evaluate the case: %s", case)
            return BenchmarkCaseResponse(
                response=None, extras=None, exit_code=ExitCode.FAILED
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run = create_run_from_jsonl(
        agent=CriteriaAgent(),
        jsonl_file="data/criteria_benchmark.jsonl",
    )

    runner = CriteriaBenchmarkRunner(
        runs=run,
        evaluator=Evaluator.LLM_CRITERIA_JUDGE_GPT_4_0,
        results_folder="data/results",
        repeats=3,
        parallel=False,
    )

    asyncio.run(runner.run())
