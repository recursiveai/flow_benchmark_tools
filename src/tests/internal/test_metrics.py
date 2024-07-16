# Copyright 2024 Recursive AI

import numpy as np
import pytest

from recursiveai.benchmark._internal._evaluation import Evaluation
from recursiveai.benchmark._internal._metrics._benchmark_metrics import BenchmarkMetrics
from recursiveai.benchmark._internal._metrics._run_metrics import RunMetrics


@pytest.fixture
def multi_evaluations():
    return [
        Evaluation(
            evaluator="",
            query="",
            reference_answer="",
            test_answer="",
            evaluation="",
            ratings=[5],
            rating_min=0,
            rating_max=10,
        ),
        None,
        Evaluation(
            evaluator="",
            query="",
            reference_answer="",
            test_answer="",
            evaluation="",
            ratings=[9],
            rating_min=0,
            rating_max=10,
        ),
    ]


@pytest.fixture
def benchmark_metrics(multi_evaluations):
    return BenchmarkMetrics(evals=multi_evaluations)


@pytest.fixture
def multi_benchmark_metrics():
    return [
        BenchmarkMetrics(
            evals=[
                Evaluation(
                    evaluator="",
                    query="",
                    reference_answer="",
                    test_answer="",
                    evaluation="",
                    ratings=[5],
                    rating_min=0,
                    rating_max=10,
                )
            ]
        ),
        BenchmarkMetrics(
            evals=[
                Evaluation(
                    evaluator="",
                    query="",
                    reference_answer="",
                    test_answer="",
                    evaluation="",
                    ratings=[9],
                    rating_min=0,
                    rating_max=10,
                )
            ]
        ),
        BenchmarkMetrics(
            evals=[
                Evaluation(
                    evaluator="",
                    query="",
                    reference_answer="",
                    test_answer="",
                    evaluation="",
                    ratings=[7],
                    rating_min=0,
                    rating_max=10,
                )
            ]
        ),
        BenchmarkMetrics(
            evals=[
                Evaluation(
                    evaluator="",
                    query="",
                    reference_answer="",
                    test_answer="",
                    evaluation="",
                    ratings=[None],
                    rating_min=0,
                    rating_max=10,
                )
            ]
        ),
    ]


@pytest.fixture
def run_metrics(multi_benchmark_metrics):
    return RunMetrics(benchmark_metrics=multi_benchmark_metrics)


def test_benchmark_metrics_num_evals(benchmark_metrics):
    assert benchmark_metrics.num_evals == 3


def test_benchmark_metrics_num_valid_ratings(benchmark_metrics):
    assert benchmark_metrics.num_valid_ratings == 2


def test_benchmark_metrics_valid_ratings(benchmark_metrics):
    assert benchmark_metrics.valid_ratings == [5, 9]


def test_benchmark_metrics_mean_rating(benchmark_metrics):
    assert np.isclose(benchmark_metrics.mean_rating, 7.0)


def test_benchmark_metrics_mean_rating_empty():
    benchmark_metrics = BenchmarkMetrics(evals=[])
    assert benchmark_metrics.mean_rating is None


def test_benchmark_metrics_std_dev(benchmark_metrics):
    assert np.isclose(benchmark_metrics.std_dev, 2.0)


def test_benchmark_metrics_std_dev_empty():
    benchmark_metrics = BenchmarkMetrics(evals=[])
    assert benchmark_metrics.std_dev is None


def test_run_metrics_num_benchmarks(run_metrics):
    assert run_metrics.num_benchmarks == 4


def test_run_metrics_ratings(run_metrics):
    assert np.all(np.isclose(run_metrics.ratings[:-1], [5.0, 9.0, 7.0]))
    assert run_metrics.ratings[3] is None


def test_run_metrics_num_valid_ratings(run_metrics):
    assert run_metrics.num_valid_ratings == 3


def test_run_metrics_valid_ratings(run_metrics):
    assert np.all(np.isclose(run_metrics.valid_ratings, [5.0, 9.0, 7.0]))


def test_run_metrics_sorted_ratings(run_metrics):
    assert run_metrics.sorted_ratings[0] is None
    assert np.all(np.isclose(run_metrics.sorted_ratings[1:], [5.0, 7.0, 9.0]))


def test_run_metrics_sorted_enumerated_ratings(run_metrics):
    keys = [key for key, _ in run_metrics.sorted_enumerated_ratings]
    values = [value for _, value in run_metrics.sorted_enumerated_ratings]
    assert values[0] is None
    assert np.all(np.isclose(values[1:], [5.0, 7.0, 9.0]))
    assert keys == [3, 0, 2, 1]


def test_run_metrics_histogram(run_metrics):
    hist = run_metrics.histogram
    assert len(hist) == 4
    assert "invalid" in hist and hist["invalid"] == 1
    assert "poor" in hist and hist["poor"] == 0
    assert "fair" in hist and hist["fair"] == 2
    assert "good" in hist and hist["good"] == 1


def test_run_metrics_mean_rating(run_metrics):
    assert np.isclose(run_metrics.mean_rating, 7.0)


def test_run_metrics_mean_rating_empty():
    run_metrics = RunMetrics(benchmark_metrics=[])
    assert run_metrics.mean_rating is None


def test_run_metrics_std_dev(run_metrics):
    assert np.isclose(run_metrics.std_dev, 1.6, atol=0.1)


def test_run_metrics_std_dev_empty():
    run_metrics = RunMetrics(benchmark_metrics=[])
    assert run_metrics.std_dev is None
