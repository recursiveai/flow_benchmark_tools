# Copyright 2024 Recursive AI

from unittest.mock import AsyncMock, Mock

import pytest

from recursiveai.benchmark._internal._evaluation import Evaluation
from recursiveai.benchmark._internal._evaluators import get_evaluator
from recursiveai.benchmark._internal._evaluators._happy import HappyEvaluator
from recursiveai.benchmark._internal._evaluators._llm_judge import LLMJudgeEvaluator
from recursiveai.benchmark._internal._evaluators._llm_jury import LLMJuryEvaluator
from recursiveai.benchmark._internal._evaluators._regex_match import RegexMatchEvaluator
from recursiveai.benchmark._internal._evaluators._strict_match import (
    StrictMatchEvaluator,
)
from recursiveai.benchmark.api.benchmark_evaluator import Evaluator


@pytest.fixture
def model_mock():
    model = Mock()
    model.name = "mock"
    return model


@pytest.fixture
def judges_mock(sample_evaluation: Evaluation):
    judges = [Mock(), Mock(), Mock()]
    judges[0].evaluate = AsyncMock(return_value=sample_evaluation)
    judges[0].llm_model = "mock"
    judges[1].evaluate = AsyncMock(return_value=sample_evaluation)
    judges[1].llm_model = "mock"
    judges[2].evaluate = AsyncMock(return_value=sample_evaluation)
    judges[2].llm_model = "mock"
    return judges


def test_valid_rating(sample_evaluation: Evaluation):
    assert sample_evaluation.rating == 7


def test_invalid_rating():
    with pytest.raises(ValueError):
        Evaluation(
            evaluator="",
            query="",
            reference_answer="",
            test_answer="",
            evaluation="",
            ratings=[6],
            rating_min=0,
            rating_max=5,
        )


def test_none_rating():
    evaluation = Evaluation(
        evaluator="",
        query="",
        reference_answer="",
        test_answer="",
        evaluation=None,
        ratings=[None],
        rating_min=0,
        rating_max=5,
    )

    assert evaluation.rating is None


def test_multiple_valid_ratings():
    evaluation = Evaluation(
        evaluator="",
        query="",
        reference_answer="",
        test_answer="",
        evaluation="",
        ratings=[3, 4, 4],
        rating_min=0,
        rating_max=5,
    )

    assert evaluation.rating == 4


def test_multiple_ratings_w_invalid():
    evaluation = Evaluation(
        evaluator="",
        query="",
        reference_answer="",
        test_answer="",
        evaluation="",
        ratings=[3, 5, None],
        rating_min=0,
        rating_max=5,
    )

    assert evaluation.rating == 4


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


def test_get_azure_gpt():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_AZURE_GPT)
    assert isinstance(evaluator, LLMJudgeEvaluator)


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


def test_get_claude_3_5_sonnet_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_CLAUDE_3_5_SONNET)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "claude-3-5-sonnet-20240620"


def test_get_claude_3_haiku_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_CLAUDE_3_HAIKU)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "claude-3-haiku-20240307"


def test_get_gemini_1_5_flash_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_GEMINI_1_5_FLASH)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "gemini-1.5-flash"


def test_get_gemini_1_5_pro_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JUDGE_GEMINI_1_5_PRO)
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator._model.name == "gemini-1.5-pro"


def test_get_gpt_claude_gemini_high_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JURY_GPT_CLAUDE_GEMINI_HIGH)
    assert isinstance(evaluator, LLMJuryEvaluator)
    for judge in evaluator._judges:
        assert isinstance(judge, LLMJudgeEvaluator)


def test_get_gpt_claude_gemini_low_evaluator():
    evaluator = get_evaluator(Evaluator.LLM_JURY_GPT_CLAUDE_GEMINI_LOW)
    assert isinstance(evaluator, LLMJuryEvaluator)
    for judge in evaluator._judges:
        assert isinstance(judge, LLMJudgeEvaluator)


def test_get_strict_match_evaluator():
    evaluator = get_evaluator(Evaluator.STRICT_MATCH)
    assert isinstance(evaluator, StrictMatchEvaluator)


def test_get_regex_match_evaluator():
    evaluator = get_evaluator(Evaluator.REGEX_MATCH)
    assert isinstance(evaluator, RegexMatchEvaluator)


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


@pytest.mark.asyncio
async def test_llm_jury_evaluate_success(judges_mock, sample_evaluation):
    jury = LLMJuryEvaluator(judge_models=[])
    jury._judges = judges_mock
    evaluation = await jury.evaluate(query="", reference_answer="", test_answer="")

    assert len(evaluation.evaluation) == 3
    assert len(evaluation.ratings) == 3
    assert evaluation.ratings == [
        sample_evaluation.rating,
        sample_evaluation.rating,
        sample_evaluation.rating,
    ]
    assert evaluation.rating == sample_evaluation.rating


@pytest.mark.asyncio
async def test_strict_match_evaluator_success():
    evaluator = StrictMatchEvaluator()
    evaluation = await evaluator.evaluate(
        query="query", reference_answer="answer", test_answer="answer"
    )
    assert evaluation.rating == 10


@pytest.mark.asyncio
async def test_strict_match_evaluator_failure():
    evaluator = StrictMatchEvaluator()
    evaluation = await evaluator.evaluate(
        query="query", reference_answer="answer", test_answer="Answer"
    )
    assert evaluation.rating == 1


@pytest.mark.asyncio
async def test_regex_match_evaluate_single_word():
    evaluator = RegexMatchEvaluator()

    evaluation = await evaluator.evaluate(
        query="query", reference_answer="answer", test_answer="answer"
    )
    assert evaluation.rating == 10

    evaluation = await evaluator.evaluate(
        query="query", reference_answer="answer", test_answer="Answer"
    )
    assert evaluation.rating == 1


@pytest.mark.asyncio
async def test_regex_match_evaluate_multiple_words():
    evaluator = RegexMatchEvaluator()

    evaluation = await evaluator.evaluate(
        query="query", reference_answer="a|b|c", test_answer="c"
    )
    assert evaluation.rating == 10

    evaluation = await evaluator.evaluate(
        query="query", reference_answer="a|b|c", test_answer="d"
    )
    assert evaluation.rating == 1


@pytest.mark.asyncio
async def test_regex_match_evaluate_number():
    evaluator = RegexMatchEvaluator()

    evaluation = await evaluator.evaluate(
        query="query", reference_answer="\\d+", test_answer="42"
    )
    assert evaluation.rating == 10

    evaluation = await evaluator.evaluate(
        query="query", reference_answer="\\d+", test_answer="answer"
    )
    assert evaluation.rating == 1
