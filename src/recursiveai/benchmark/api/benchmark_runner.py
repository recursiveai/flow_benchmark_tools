import asyncio
import datetime
import json
import logging
import os
import time

from .._internal._benchmark_output import BenchmarkOutput
from .._internal._evaluators import get_evaluator
from .._internal._run_output import RunOutput
from .benchmark import Benchmark
from .benchmark_agent import BenchmarkAgent
from .benchmark_evaluator import Evaluator
from .benchmark_run import BenchmarkRun
from .exit_code import ExitCode

_logger = logging.getLogger(__name__)

_MAX_NUM_REPEATS = 20

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

    async def run(self) -> None:
        start_time = time.time()
        results = await asyncio.gather(*[self._execute_run(run) for run in self._runs])
        runtime = time.time() - start_time
        self._save_run_results_to_json(results=results, runtime=runtime)

    async def _execute_run(self, run: BenchmarkRun) -> RunOutput:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self._parallel:
            outputs = await asyncio.gather(
                *[
                    self._execute_benchmark(
                        agent=run.agent,
                        benchmark=benchmark,
                        idx=idx,
                        total=len(run.benchmarks),
                    )
                    for idx, benchmark in enumerate(run.benchmarks)
                ]
            )
        else:
            outputs = []
            for idx, benchmark in enumerate(run.benchmarks):
                output = await self._execute_benchmark(
                    agent=run.agent,
                    benchmark=benchmark,
                    idx=idx,
                    total=len(run.benchmarks),
                )
                outputs.append(output)
        return RunOutput(
            date=date, agent_name=run.agent.name, benchmark_outputs=outputs
        )

    async def _execute_benchmark(
        self, agent: BenchmarkAgent, benchmark: Benchmark, idx: int, total: int
    ) -> BenchmarkOutput:
        _logger.info(
            "Benchmark %s of %s: agent=%s benchmark=%s",
            idx + 1,
            total,
            agent.name,
            benchmark,
        )
        evaluations = []
        start_time = time.time()
        for repeat in range(self._repeats):
            _logger.info("Repeat %s of %s", repeat + 1, self._repeats)
            evaluation = None
            try:
                response = await agent.run_benchmark(benchmark)
                if response.exit_code == ExitCode.SUCCESS:
                    evaluation = await self._evaluator.evaluate(
                        query=benchmark.query,
                        reference_answer=benchmark.reference_answer,
                        test_answer=response.response,
                    )
                else:
                    _logger.error(
                        "Benchmark exit_code is not SUCCESS: %s", response.exit_code
                    )

            except Exception:
                _logger.exception("Caught exception while running benchmark")

            evaluations.append(evaluation)
        runtime = time.time() - start_time
        return BenchmarkOutput(
            id=idx,
            info=benchmark,
            repeats=self._repeats,
            evaluations=evaluations,
            runtime=runtime,
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
