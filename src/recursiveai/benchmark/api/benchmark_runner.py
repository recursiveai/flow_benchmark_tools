# Copyright 2024 Recursive AI

import asyncio
import datetime
import json
import logging
import os
import time

from .._internal._benchmark_output import BenchmarkOutput
from .._internal._evaluators import get_criteria_evaluator, get_evaluator
from .._internal._run_output import RunOutput
from .benchmark_agent import BenchmarkAgent
from .benchmark_case import BenchmarkCase
from .benchmark_evaluator import Evaluator
from .benchmark_run import BenchmarkRun
from .exit_code import ExitCode

_logger = logging.getLogger(__name__)

_MAX_NUM_REPEATS = 20
_MAX_CONCURRENT_CASES = 1000

_DEFAULT_RESULTS_FOLDER = "benchmark/results/"


class BenchmarkRunner:
    def __init__(
        self,
        runs: list[BenchmarkRun] | BenchmarkRun,
        evaluator: Evaluator = Evaluator.LLM_JUDGE_GPT_4_0,
        results_folder=_DEFAULT_RESULTS_FOLDER,
        results_file="",
        repeats: int = 1,
        parallel: bool = False,
        max_concurrency: int = _MAX_CONCURRENT_CASES,
    ) -> None:
        if isinstance(runs, list):
            self._runs = runs
        else:
            self._runs = [runs]
        self._evaluator = get_evaluator(evaluator=evaluator)
        self._results_folder = results_folder
        self._results_file = results_file
        if repeats < 1:
            self._repeats = 1
        elif repeats > _MAX_NUM_REPEATS:
            _logger.warning(
                "Requested repeats:%s exceeds maximum allowed:%s",
                repeats,
                _MAX_NUM_REPEATS,
            )
            self._repeats = _MAX_NUM_REPEATS
        else:
            self._repeats = repeats
        self._parallel = parallel
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self) -> None:
        start_time = time.time()
        results = await asyncio.gather(*[self._execute_run(run) for run in self._runs])
        runtime = time.time() - start_time
        self._save_run_results_to_json(results=results, runtime=runtime)

    async def _execute_run(self, run: BenchmarkRun) -> RunOutput:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        await run.agent.before_run(run.benchmark)
        cases = run.benchmark.cases
        if self._parallel:
            outputs = await asyncio.gather(
                *[
                    self._execute_benchmark_case(
                        agent=run.agent,
                        case=case,
                        idx=idx,
                        total=len(cases),
                    )
                    for idx, case in enumerate(cases)
                ]
            )
        else:
            outputs = []
            for idx, case in enumerate(cases):
                output = await self._execute_benchmark_case(
                    agent=run.agent,
                    case=case,
                    idx=idx,
                    total=len(cases),
                )
                outputs.append(output)
        await run.agent.after_run(run.benchmark)
        return RunOutput(
            date=date, agent_name=run.agent.name, benchmark_outputs=outputs
        )

    async def _execute_benchmark_case(
        self, agent: BenchmarkAgent, case: BenchmarkCase, idx: int, total: int
    ) -> BenchmarkOutput:
        async with self._semaphore:
            _logger.info(
                "Benchmark %s of %s: agent=%s benchmark=%s",
                idx + 1,
                total,
                agent.name,
                case,
            )
            evaluations = []
            case_runtimes = []
            start_time = time.time()
            for repeat in range(self._repeats):
                _logger.info("Repeat %s of %s", repeat + 1, self._repeats)
                evaluation = None
                try:
                    await agent.before_case(case)
                    case_start_time = time.time()
                    response = await agent.run_benchmark_case(case)
                    case_end_time = time.time()
                    if response.exit_code == ExitCode.SUCCESS:
                        evaluation = await self._evaluator.evaluate(
                            query=case.query,
                            reference_answer=case.reference_answer,
                            test_answer=response.response,
                        )
                    else:
                        _logger.error(
                            "Benchmark exit_code is not SUCCESS: %s", response.exit_code
                        )

                except Exception:
                    _logger.exception("Caught exception while running benchmark")

                else:
                    if response.exit_code == ExitCode.SUCCESS:
                        case_runtimes.append(case_end_time - case_start_time)

                finally:
                    try:
                        await agent.after_case(case)
                    except Exception:
                        _logger.exception(
                            "Caught exception while running after_benchmark"
                        )

                evaluations.append(evaluation)
            total_runtime = time.time() - start_time

            mean_case_runtime = None
            if case_runtimes:
                mean_case_runtime = sum(case_runtimes) / len(case_runtimes)

            return BenchmarkOutput(
                id=idx,
                info=case,
                repeats=self._repeats,
                evaluations=evaluations,
                mean_case_runtime=mean_case_runtime,
                total_runtime=total_runtime,
            )

    def _save_run_results_to_json(
        self, results: list[RunOutput], runtime: float | None = None
    ) -> None:
        filename = self._results_file
        if not self._results_file:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"benchmark_run_{date}.json"

        folder = self._results_folder
        os.makedirs(folder, exist_ok=True)

        full_path = os.path.join(folder, filename)
        _logger.info("Saving results to %s", full_path)
        output = {}
        with open(full_path, "w") as f:
            output["total_runtime"] = runtime
            output["runs"] = [result.model_dump() for result in results]
            json.dump(output, f, ensure_ascii=False, indent=4)


class CriteriaBenchmarkRunner(BenchmarkRunner):
    """Custom BenchmarkRunner, where we do not have a reference_answer to evaluate.
    Instead, we use a criteria-based evaluator to get subjective scores of the 'query'
    based on the 'criteria' defined in the 'extra' dict of the BenchmarkCase.
    """

    def __init__(
        self,
        runs: list[BenchmarkRun] | BenchmarkRun,
        evaluator: Evaluator = Evaluator.LLM_CRITERIA_JUDGE_GPT_4_0,
        results_folder=_DEFAULT_RESULTS_FOLDER,
        results_file="",
        repeats: int = 1,
        parallel: bool = False,
        max_concurrency: int = _MAX_CONCURRENT_CASES,
    ) -> None:
        super().__init__(
            runs=runs,
            evaluator=evaluator,
            results_folder=results_folder,
            results_file=results_file,
            repeats=repeats,
            parallel=parallel,
            max_concurrency=max_concurrency,
        )
        self._evaluator = get_criteria_evaluator(evaluator=evaluator)

    async def _execute_benchmark_case(
        self, agent: BenchmarkAgent, case: BenchmarkCase, idx: int, total: int
    ) -> BenchmarkOutput:
        async with self._semaphore:
            _logger.info(
                "Benchmark %s of %s: agent=%s benchmark=%s",
                idx + 1,
                total,
                agent.name,
                case,
            )
            evaluations = []
            case_runtimes = []
            start_time = time.time()
            for repeat in range(self._repeats):
                _logger.info("Repeat %s of %s", repeat + 1, self._repeats)
                evaluation = None
                try:
                    await agent.before_case(case)
                    case_start_time = time.time()
                    response = await agent.run_benchmark_case(case)

                    case_end_time = time.time()
                    if response.exit_code == ExitCode.SUCCESS:
                        evaluation = await self._evaluator.evaluate(
                            criteria=case.extras["criteria"], test_text=case.query
                        )
                    else:
                        _logger.error(
                            "Benchmark exit_code is not SUCCESS: %s", response.exit_code
                        )

                except Exception:
                    _logger.exception("Caught exception while running benchmark")

                else:
                    if response.exit_code == ExitCode.SUCCESS:
                        case_runtimes.append(case_end_time - case_start_time)

                finally:
                    try:
                        await agent.after_case(case)
                    except Exception:
                        _logger.exception(
                            "Caught exception while running after_benchmark"
                        )

                evaluations.append(evaluation)
            total_runtime = time.time() - start_time

            mean_case_runtime = None
            if case_runtimes:
                mean_case_runtime = sum(case_runtimes) / len(case_runtimes)

            return BenchmarkOutput(
                id=idx,
                info=case,
                repeats=self._repeats,
                evaluations=evaluations,
                mean_case_runtime=mean_case_runtime,
                total_runtime=total_runtime,
            )
