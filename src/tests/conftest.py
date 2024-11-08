# Copyright 2024 Recursive AI

import pytest

from recursiveai.benchmark._internal._evaluation import Evaluation

DEFAULT_TEST_RATING = 7


@pytest.fixture
def sample_evaluation() -> Evaluation:
    return Evaluation(
        evaluator="test_evaluator",
        query="test_query",
        reference_answer="test_reference_answer",
        test_answer="test_test_answer",
        evaluation="test_evaluation",
        ratings=[DEFAULT_TEST_RATING],
        rating_min=0,
        rating_max=10,
    )
