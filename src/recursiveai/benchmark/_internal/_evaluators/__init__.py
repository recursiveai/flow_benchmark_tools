# Copyright 2024 Recursive AI

from recursiveai.benchmark.api.benchmark_evaluator import Evaluator

from .._benchmark_evaluator import BenchmarkEvaluator
from .._llm._openai_gpt_model import GPT_3_5_TURBO, GPT_4_O, GPT_4_TURBO_PREVIEW
from ._happy import HappyEvaluator
from ._llm_judge import LLMJudgeEvaluator


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
