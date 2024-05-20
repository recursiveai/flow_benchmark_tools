import datetime
import json
import os

import pytest

from recursiveai.benchmark._internal._benchmark_output import BenchmarkOutput
from recursiveai.benchmark._internal._run_output import RunOutput
from recursiveai.benchmark.api import BenchmarkRunner
from recursiveai.benchmark.api.benchmark_runner import _MAX_NUM_REPEATS


@pytest.fixture
def benchmark_outputs(benchmark_list, sample_evaluation):
    benchmark_outputs = []
    for bm in benchmark_list:
        output = BenchmarkOutput(
            info=bm,
            repeats=2,
            evaluations=[
                sample_evaluation,
                None,
            ],
        )
        benchmark_outputs.append(output)
    return benchmark_outputs


@pytest.fixture
def run_outputs(benchmark_outputs):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return [
        RunOutput(
            date=date,
            agent_name="test_agent_name_0",
            benchmark_outputs=[],
        ),
        RunOutput(
            date=date,
            agent_name="test_agent_name_1",
            benchmark_outputs=benchmark_outputs,
        ),
    ]


def test_save_runs_to_json(run_outputs):
    runner = BenchmarkRunner(
        runs=[], results_folder="benchmark_temp", results_file="results.json"
    )
    try:
        runner._save_run_results_to_json(results=run_outputs)

        # If the output is not a valid JSON, this will raise an exception and the test will fail
        with open("benchmark_temp/results.json", "r") as f:
            json.load(f)
    finally:
        try:
            os.remove("benchmark_temp/results.json")
            os.rmdir("benchmark_temp")
        except OSError:
            pass


@pytest.mark.asyncio
async def test_execute_run(benchmark_run, callback_agent, benchmark_list):
    runner = BenchmarkRunner(runs=[])
    result = await runner._execute_run(run=benchmark_run)

    assert result.agent_name == callback_agent.name
    assert len(result.benchmark_outputs) == len(benchmark_list)
    for idx, out in enumerate(result.benchmark_outputs):
        assert out.info == benchmark_list[idx]
        assert len(out.evaluations) == out.repeats


def test_negative_repeats():
    runner = BenchmarkRunner(runs=[], repeats=-10)
    assert runner._repeats == 1


def test_exceed_max_repeats():
    runner = BenchmarkRunner(runs=[], repeats=1000)
    assert runner._repeats == _MAX_NUM_REPEATS
