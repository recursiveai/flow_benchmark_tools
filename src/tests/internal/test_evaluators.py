import pytest

from recursiveai.benchmark._internal._evaluation import Evaluation
from recursiveai.benchmark._internal._evaluators._happy_evaluator import HappyEvaluator


def test_valid_rating():
    evaluation = Evaluation(
        evaluator="",
        query="",
        reference_answer="",
        test_answer="",
        evaluation="",
        rating=7,
        rating_min=0,
        rating_max=10,
    )

    assert evaluation.rating == 7


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
