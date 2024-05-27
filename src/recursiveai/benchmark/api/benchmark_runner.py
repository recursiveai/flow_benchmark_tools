import asyncio
import datetime
import json
import logging
import os

from .._internal._benchmark_output import BenchmarkOutput
from .._internal._evaluators import get_evaluator
from .._internal._run_output import RunOutput
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
        evaluator: Evaluator = Evaluator.LLM_JUDGE_GPT_4_TURBO_PREVIEW,
        results_folder=_DEFAULT_RESULTS_FOLDER,
        results_file="",
        repeats: int = 1,
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
                f"Requested repeats:{repeats} exceeds maximum allowed:{_MAX_NUM_REPEATS}"
            )
            self._repeats = _MAX_NUM_REPEATS
        else:
            self._repeats = repeats

    async def run(self) -> None:
        results = await asyncio.gather(*[self._execute_run(run) for run in self._runs])
        self._save_run_results_to_json(results=results)

    async def _execute_run(self, run: BenchmarkRun) -> RunOutput:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outputs = []
        for idx, benchmark in enumerate(run.benchmarks):
            _logger.info(
                f"Benchmark {idx+1} of {len(run.benchmarks)}: agent={run.agent.name} benchmark={benchmark}"
            )
            evaluations = []
            for repeat in range(self._repeats):
                _logger.info(f"Repeat {repeat+1} of {self._repeats}")
                evaluation = None
                try:
                    response = await run.agent.run_benchmark(benchmark)
                    if response.exit_code == ExitCode.SUCCESS:
                        evaluation = await self._evaluator.evaluate(
                            query=benchmark.query,
                            reference_answer=benchmark.reference_answer,
                            test_answer=response.response,
                        )
                    else:
                        _logger.error(
                            f"Benchmark exit_code is not SUCCESS: {response.exit_code}"
                        )

                except Exception:
                    _logger.exception("Caught exception while running benchmark")

                evaluations.append(evaluation)
            outputs.append(
                BenchmarkOutput(
                    info=benchmark, repeats=self._repeats, evaluations=evaluations
                )
            )
        return RunOutput(
            date=date, agent_name=run.agent.name, benchmark_outputs=outputs
        )

    def _save_run_results_to_json(
        self,
        results: list[RunOutput],
    ) -> None:
        filename = self._results_file
        if not self._results_file:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"benchmark_run_{date}.json"

        folder = self._results_folder
        os.makedirs(folder, exist_ok=True)

        full_path = os.path.join(folder, filename)
        _logger.info(f"Saving results to {full_path}")
        with open(full_path, "w") as f:
            dumps = {"runs": [result.model_dump() for result in results]}
            json.dump(dumps, f, ensure_ascii=False, indent=4)
