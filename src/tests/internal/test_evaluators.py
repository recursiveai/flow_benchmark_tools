from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._evaluation import Evaluation
from recursiveai.benchmark._internal._evaluators import get_evaluator
from recursiveai.benchmark._internal._evaluators._happy import HappyEvaluator
from recursiveai.benchmark._internal._evaluators._llm_judge import LLMJudgeEvaluator
from recursiveai.benchmark.api.benchmark_evaluator import Evaluator


@pytest.fixture
def model_mock():
    model = Mock()
    model.name = "mock"
    return model


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


def test_get_happy_evaluator():
    evaluator = get_evaluator(Evaluator.HAPPY)
    assert isinstance(evaluator, HappyEvaluator)


def test_get_gpt_3_5_turbo_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_GPT_3_5_TURBO)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "gpt-3.5-turbo"


def test_get_gpt_4_turbo_preview_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_GPT_4_TURBO_PREVIEW)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "gpt-4-turbo-preview"


def test_get_gpt_4_o_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_GPT_4_0)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "gpt-4o"


def test_get_claude_3_opus_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_CLAUDE_3_OPUS)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "claude-3-opus-20240229"


def test_get_claude_3_haiku_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_CLAUDE_3_HAIKU)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "claude-3-haiku-20240307"


def test_get_default_evaluator():
    evaluator = get_evaluator("test")
    assert isinstance(evaluator, LLMJudgeEvaluator)


@pytest.mark.asyncio
async def test_llm_judge_evaluate_success(model_mock):
    mock_evaluation = "Rating: [[5]]"
    model_mock.async_chat_completion = AsyncMock(return_value=mock_evaluation)
    evaluator = LLMJudgeEvaluator(model=model_mock)
    evaluation = await evaluator.evaluate(query="", reference_answer="", test_answer="")
    assert evaluation.rating == 5
    assert evaluation.evaluation == mock_evaluation


@pytest.mark.asyncio
async def test_llm_judge_evaluate_no_rating(model_mock):
    mock_evaluation = "No rating"
    model_mock.async_chat_completion = AsyncMock(return_value=mock_evaluation)
    evaluator = LLMJudgeEvaluator(model=model_mock)
    evaluation = await evaluator.evaluate(query="", reference_answer="", test_answer="")
    assert evaluation.rating == None
    assert evaluation.evaluation == mock_evaluation


@pytest.mark.asyncio
async def test_llm_judge_evaluate_none(model_mock):
    model_mock.async_chat_completion = AsyncMock(return_value=None)
    evaluator = LLMJudgeEvaluator(model=model_mock)
    evaluation = await evaluator.evaluate(query="", reference_answer="", test_answer="")
    assert evaluation.rating == None
    assert evaluation.evaluation == None
