# Copyright 2024 Recursive AI

from enum import Enum


class Evaluator(str, Enum):
    HAPPY = "happy"

    LLM_JUDGE_GPT_3_5_TURBO = "llm_judge_gpt-3.5-turbo"
    LLM_JUDGE_GPT_4_TURBO_PREVIEW = "llm_judge_gpt-4-turbo-preview"
    LLM_JUDGE_GPT_4_0 = "llm_judge_gpt-4o"
    LLM_JUDGE_CLAUDE_3_OPUS = "llm_judge_claude-3-opus"
    LLM_JUDGE_CLAUDE_3_5_SONNET = "llm_judge_claude-3-5-sonnet"
    LLM_JUDGE_CLAUDE_3_HAIKU = "llm_judge_claude-3-haiku"
    LLM_JUDGE_GEMINI_1_5_FLASH = "llm_judge_gemini-1.5-flash"
    LLM_JUDGE_GEMINI_1_5_PRO = "llm_judge_gemini-1.5-pro"
    LLM_JUDGE_AZURE_GPT = "llm_judge_azure-gpt"

    LLM_JURY_GPT_CLAUDE_GEMINI_HIGH = "llm_jury_gpt_claude_gemini_high"
    LLM_JURY_GPT_CLAUDE_GEMINI_LOW = "llm_jury_gpt_claude_gemini_low"

    LLM_CRITERIA_JUDGE_GPT_4_0 = "llm_criteria_judge_gpt-4o"

    LLM_CRITERIA_JURY_GPT_CLAUDE_GEMINI_HIGH = (
        "llm_criteria_jury_gpt_claude_gemini_high"
    )
    LLM_CRITERIA_JURY_GPT_CLAUDE_GEMINI_LOW = "llm_criteria_jury_gpt_claude_gemini_low"

    STRICT_MATCH = "strict_match"
    REGEX_MATCH = "regex_match"
