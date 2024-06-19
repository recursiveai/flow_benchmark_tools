# Copyright 2024 Recursive AI

from .._benchmark_evaluator import BenchmarkEvaluator
from .._llm._anthropic_claude_model import CLAUDE_3_HAIKU, CLAUDE_3_OPUS
from .._llm._google_gemini_model import GEMINI_1_5_FLASH, GEMINI_1_5_PRO
from .._llm._openai_gpt_model import GPT_3_5_TURBO, GPT_4_O, GPT_4_TURBO_PREVIEW
from ._happy import HappyEvaluator
from ._llm_judge import LLMJudgeEvaluator


def get_evaluator(evaluator: str) -> BenchmarkEvaluator:
    match (evaluator):
        case "happy":
            return HappyEvaluator()
        case "llm_judge_gpt-3.5-turbo":
            return LLMJudgeEvaluator(model=GPT_3_5_TURBO)
        case "llm_judge_gpt-4-turbo-preview":
            return LLMJudgeEvaluator(model=GPT_4_TURBO_PREVIEW)
        case "llm_judge_gpt-4o":
            return LLMJudgeEvaluator(model=GPT_4_O)
        case "llm_judge_claude-3-opus":
            return LLMJudgeEvaluator(model=CLAUDE_3_OPUS)
        case "llm_judge_claude-3-haiku":
            return LLMJudgeEvaluator(model=CLAUDE_3_HAIKU)
        case "llm_judge_gemini-1.5-flash":
            return LLMJudgeEvaluator(model=GEMINI_1_5_FLASH)
        case "llm_judge_gemini-1.5-pro":
            return LLMJudgeEvaluator(model=GEMINI_1_5_PRO)
        case _:
            return LLMJudgeEvaluator(model=GPT_4_O)
