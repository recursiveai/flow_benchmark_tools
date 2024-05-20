import pytest

from recursiveai.benchmark._internal._evaluation import Evaluation
from recursiveai.benchmark._internal._evaluators._happy_evaluator import HappyEvaluator
from recursiveai.benchmark.api.benchmark_evaluator import Evaluator, get_evaluator


def test_valid_rating(sample_evaluation):
    assert sample_evaluation.rating == 7


def test_invalid_rating():
    with pytest.raises(ValueError):
        Evaluation(
            evaluator="",
            query="",
            reference_answer="",
            test_answer="",
            evaluation="",
            rating=6,
            rating_min=0,
            rating_max=5,
        )


def test_none_rating():
    evaluation = Evaluation(
        evaluator="",
        query="",
        reference_answer="",
        test_answer="",
        evaluation="",
        rating=None,
        rating_min=0,
        rating_max=5,
    )

    assert evaluation.rating is None


@pytest.mark.asyncio
async def test_happy_evaluator():
    rating = 8
    evaluator = HappyEvaluator(rating=rating)
    evaluation = await evaluator.evaluate(query="", reference_answer="", test_answer="")
    assert evaluation.rating == rating
    assert evaluation.evaluator == evaluator.name


def test_get_default_evaluator():
    evaluator = get_evaluator("")
    assert isinstance(evaluator, HappyEvaluator)


def test_get_happy_evaluator():
    evaluator = get_evaluator(Evaluator.HAPPY)
    assert isinstance(evaluator, HappyEvaluator)
