from enum import Enum

from recursiveai.benchmark._internal._benchmark_evaluator import BenchmarkEvaluator
from recursiveai.benchmark._internal._evaluators._happy import HappyEvaluator
from recursiveai.benchmark._internal._evaluators._llm_judge import LLMJudgeEvaluator
from recursiveai.benchmark._internal._llm._openai_gpt_model import (
    GPT_3_5_TURBO,
    GPT_4_O,
    GPT_4_TURBO_PREVIEW,
)


class Evaluator(str, Enum):
    HAPPY = "happy"
    LLM_JUDGE_GPT_3_5_TURBO = "llm_judge_gpt-3.5-turbo"
    LLM_JUDGE_GPT_4_TURBO_PREVIEW = "llm_judge_gpt-4-turbo-preview"
    LLM_JUDGE_GPT_4_0 = "llm_judge_gpt-4o"


def get_evaluator(evaluator: Evaluator) -> BenchmarkEvaluator:
    match (evaluator):
        case Evaluator.HAPPY:
            return HappyEvaluator()
        case Evaluator.LLM_JUDGE_GPT_3_5_TURBO:
            return LLMJudgeEvaluator(model=GPT_3_5_TURBO)
        case Evaluator.LLM_JUDGE_GPT_4_TURBO_PREVIEW:
            return LLMJudgeEvaluator(model=GPT_4_TURBO_PREVIEW)
        case Evaluator.LLM_JUDGE_GPT_4_0:
            return LLMJudgeEvaluator(model=GPT_4_O)
        case _:
            return LLMJudgeEvaluator(model=GPT_4_O)
