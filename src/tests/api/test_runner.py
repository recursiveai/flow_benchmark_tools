# Copyright 2024 Recursive AI

import datetime
import json
import os
from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._benchmark_evaluator import BenchmarkEvaluator
from recursiveai.benchmark._internal._benchmark_output import BenchmarkOutput
from recursiveai.benchmark._internal._run_output import RunOutput
from recursiveai.benchmark.api import (
    Benchmark,
    BenchmarkCaseResponse,
    BenchmarkRun,
    BenchmarkRunner,
    ExitCode,
)
from recursiveai.benchmark.api.benchmark_evaluator import Evaluator
from recursiveai.benchmark.api.benchmark_runner import _MAX_NUM_REPEATS


@pytest.fixture
def benchmark_outputs(benchmark_case_list, sample_evaluation):
    benchmark_outputs = []
    for idx, bm in enumerate(benchmark_case_list):
        output = BenchmarkOutput(
            id=idx,
            info=bm,
            repeats=2,
            evaluations=[
                sample_evaluation,
                None,
            ],
            mean_case_runtime=0.3,
            total_runtime=1.0,
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
        runner._save_run_results_to_json(results=run_outputs, runtime=3.5)

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
async def test_execute_run_sequential_success(benchmark_case_list):
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.run_benchmark_case = AsyncMock(
        return_value=BenchmarkCaseResponse(
            exit_code=ExitCode.SUCCESS, response="success"
        )
    )
    run = BenchmarkRun(agent=agent, benchmark=Benchmark(cases=benchmark_case_list))

    runner = BenchmarkRunner(runs=[], evaluator=Evaluator.HAPPY, parallel=False)
    result = await runner._execute_run(run=run)

    assert len(result.benchmark_outputs) == len(benchmark_case_list)
    agent.before_run.assert_awaited_once()
    agent.after_run.assert_awaited_once()
    for idx, out in enumerate(result.benchmark_outputs):
        assert out.id == idx
        assert out.info == benchmark_case_list[idx]
        assert len(out.evaluations) == out.repeats
        assert all([evl.test_answer == "success" for evl in out.evaluations])
        assert out.total_runtime is not None and out.total_runtime > 0.0
        assert out.mean_case_runtime is not None and out.mean_case_runtime > 0.0
        agent.before_case.assert_any_await(benchmark_case_list[idx])
        agent.after_case.assert_any_await(benchmark_case_list[idx])


@pytest.mark.asyncio
async def test_execute_run_parallel_success(benchmark_case_list):
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.run_benchmark_case = AsyncMock(
        return_value=BenchmarkCaseResponse(
            exit_code=ExitCode.SUCCESS, response="success"
        )
    )
    run = BenchmarkRun(agent=agent, benchmark=Benchmark(cases=benchmark_case_list))

    runner = BenchmarkRunner(runs=[], evaluator=Evaluator.HAPPY, parallel=True)
    result = await runner._execute_run(run=run)

    assert len(result.benchmark_outputs) == len(benchmark_case_list)
    agent.before_run.assert_awaited_once()
    agent.after_run.assert_awaited_once()
    for idx, out in enumerate(result.benchmark_outputs):
        assert out.id == idx
        assert out.info == benchmark_case_list[idx]
        assert len(out.evaluations) == out.repeats
        assert all([evl.test_answer == "success" for evl in out.evaluations])
        assert out.total_runtime is not None and out.total_runtime > 0.0
        assert out.mean_case_runtime is not None and out.mean_case_runtime > 0.0
        agent.before_case.assert_any_await(benchmark_case_list[idx])
        agent.after_case.assert_any_await(benchmark_case_list[idx])


@pytest.mark.asyncio
async def test_execute_run_failure_exit_code(benchmark_case_list):
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.run_benchmark_case = AsyncMock(
        return_value=BenchmarkCaseResponse(exit_code=ExitCode.FAILED)
    )
    run = BenchmarkRun(agent=agent, benchmark=Benchmark(cases=benchmark_case_list))

    runner = BenchmarkRunner(runs=[], evaluator=Evaluator.HAPPY)
    result = await runner._execute_run(run=run)

    assert len(result.benchmark_outputs) == len(benchmark_case_list)
    agent.before_run.assert_awaited_once()
    agent.after_run.assert_awaited_once()
    for idx, out in enumerate(result.benchmark_outputs):
        assert out.id == idx
        assert out.info == benchmark_case_list[idx]
        assert len(out.evaluations) == out.repeats
        assert all([evl is None for evl in out.evaluations])
        assert out.total_runtime is not None and out.total_runtime > 0.0
        assert out.mean_case_runtime is None
        agent.before_case.assert_any_await(benchmark_case_list[idx])
        agent.after_case.assert_any_await(benchmark_case_list[idx])


@pytest.mark.asyncio
async def test_execute_run_failure_exception(benchmark_case_list):
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.run_benchmark_case = AsyncMock(side_effect=Exception())
    run = BenchmarkRun(agent=agent, benchmark=Benchmark(cases=benchmark_case_list))

    runner = BenchmarkRunner(runs=[], evaluator=Evaluator.HAPPY)
    result = await runner._execute_run(run=run)

    assert len(result.benchmark_outputs) == len(benchmark_case_list)
    agent.before_run.assert_awaited_once()
    agent.after_run.assert_awaited_once()
    for idx, out in enumerate(result.benchmark_outputs):
        assert out.id == idx
        assert out.info == benchmark_case_list[idx]
        assert len(out.evaluations) == out.repeats
        assert all([evl is None for evl in out.evaluations])
        assert out.total_runtime is not None and out.total_runtime > 0.0
        assert out.mean_case_runtime is None
        agent.before_case.assert_any_await(benchmark_case_list[idx])
        agent.after_case.assert_any_await(benchmark_case_list[idx])


def test_negative_repeats():
    runner = BenchmarkRunner(runs=[], repeats=-10)
    assert runner._repeats == 1


def test_exceed_max_repeats():
    runner = BenchmarkRunner(runs=[], repeats=1000)
    assert runner._repeats == _MAX_NUM_REPEATS


def test_single_run():
    mock_run = Mock()
    runner = BenchmarkRunner(runs=mock_run)
    assert type(runner._runs) is list
    assert runner._runs == [mock_run]


def test_evaluator_type():
    runner = BenchmarkRunner(runs=[], evaluator=Evaluator.HAPPY)
    assert isinstance(runner._evaluator, BenchmarkEvaluator)
