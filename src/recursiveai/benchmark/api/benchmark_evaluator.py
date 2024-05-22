from enum import Enum


class Evaluator(str, Enum):
    HAPPY = "happy"
    LLM_JUDGE_GPT_3_5_TURBO = "llm_judge_gpt-3.5-turbo"
    LLM_JUDGE_GPT_4_TURBO_PREVIEW = "llm_judge_gpt-4-turbo-preview"
    LLM_JUDGE_GPT_4_0 = "llm_judge_gpt-4o"
