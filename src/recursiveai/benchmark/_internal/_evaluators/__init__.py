# Copyright 2024 Recursive AI

from .._benchmark_evaluator import BenchmarkEvaluator
from .._criteria_evaluator import CriteriaEvaluator
from .._llm._anthropic_claude_model import (
    CLAUDE_3_5_SONNET,
    CLAUDE_3_HAIKU,
    CLAUDE_3_OPUS,
)
from .._llm._azure_openai_gpt_model import AZURE_GPT
from .._llm._google_gemini_model import GEMINI_1_5_FLASH, GEMINI_1_5_PRO
from .._llm._openai_gpt_model import GPT_3_5_TURBO, GPT_4_O, GPT_4_TURBO_PREVIEW
from ._happy import HappyEvaluator
from ._llm_criteria_judge import LLMCriteriaJudgeEvaluator
from ._llm_criteria_jury import LLMCriteriaJuryEvaluator
from ._llm_judge import LLMJudgeEvaluator
from ._llm_jury import LLMJuryEvaluator
from ._regex_match import RegexMatchEvaluator
from ._strict_match import StrictMatchEvaluator


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
        case "llm_judge_claude-3-5-sonnet":
            return LLMJudgeEvaluator(model=CLAUDE_3_5_SONNET)
        case "llm_judge_claude-3-haiku":
            return LLMJudgeEvaluator(model=CLAUDE_3_HAIKU)
        case "llm_judge_gemini-1.5-flash":
            return LLMJudgeEvaluator(model=GEMINI_1_5_FLASH)
        case "llm_judge_gemini-1.5-pro":
            return LLMJudgeEvaluator(model=GEMINI_1_5_PRO)
        case "llm_judge_azure-gpt":
            return LLMJudgeEvaluator(model=AZURE_GPT)
        case "llm_jury_gpt_claude_gemini_high":
            return LLMJuryEvaluator(
                judge_models=[GPT_4_O, CLAUDE_3_5_SONNET, GEMINI_1_5_PRO]
            )
        case "llm_jury_gpt_claude_gemini_low":
            return LLMJuryEvaluator(
                judge_models=[GPT_3_5_TURBO, CLAUDE_3_HAIKU, GEMINI_1_5_FLASH]
            )

        case "strict_match":
            return StrictMatchEvaluator()
        case "regex_match":
            return RegexMatchEvaluator()
        case _:
            return LLMJudgeEvaluator(model=GPT_4_O)


def get_criteria_evaluator(evaluator: str) -> CriteriaEvaluator:
    match (evaluator):
        case "llm_criteria_judge_gpt-4o":
            return LLMCriteriaJudgeEvaluator(model=GPT_4_O)
        case "llm_criteria_jury_gpt_claude_gemini_high":
            return LLMCriteriaJuryEvaluator(
                judge_models=[GPT_4_O, CLAUDE_3_5_SONNET, GEMINI_1_5_PRO]
            )
        case "llm_criteria_jury_gpt_claude_gemini_low":
            return LLMCriteriaJuryEvaluator(
                judge_models=[GPT_3_5_TURBO, CLAUDE_3_HAIKU, GEMINI_1_5_FLASH]
            )
        case _:
            return LLMCriteriaJudgeEvaluator(model=GPT_4_O)
