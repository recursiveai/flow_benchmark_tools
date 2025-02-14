# Copyright 2024 Recursive AI

import logging
import re

from .._criteria_evaluator import CriteriaEvaluator
from .._evaluation import Evaluation
from .._llm._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)

_CRITERIA_JUDGE_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of text media.
You will be provided with:
    - a text snippet to evaluate (enclosed in <text_snippet> tags)
    - some evaluation criteria (enclosed in <evaluation_criteria> tags)
Please act as an impartial and objective judge and evaluate the quality of the text snippet using the defined evalution criteria.
Begin your evaluation by providing a short description of how the text snippet does or does not meet the evaluation criteria.
After providing your short description, you must rate the test snippet on a scale of 1 to 10 by strictly following this format: '[[rating]]', for example: 'Rating: [[5]]'
If the test snippet is empty, give a rating of 1.
Write your response in English, even if the text snippet and/or the evaluation criteria are written in a different language.
"""

_CRITERIA_JUDGE_USER_PROMPT = """\
<text_snippet>
{text_snippet}
</text_snippet>

<evaluation_criteria>
{evaluation_criteria}
</evaluation_criteria>
"""


class LLMCriteriaJudgeEvaluator(CriteriaEvaluator):

    def __init__(self, model: LLMModel) -> None:
        super().__init__()
        self._model = model

    @property
    def llm_model(self) -> str:
        return self._model.name

    async def evaluate(self, criteria: str, test_text: str) -> Evaluation:

        user_prompt = _CRITERIA_JUDGE_USER_PROMPT.format(
            evaluation_criteria=criteria,
            text_snippet=test_text,
        )

        chat = [
            ChatMessage(content=_CRITERIA_JUDGE_SYSTEM_PROMPT, role="system"),
            ChatMessage(content=user_prompt, role="user"),
        ]

        evaluation = await self._model.async_chat_completion(
            chat,
            temperature=0.5,
            max_tokens=1024,
            timeout=60,
        )

        rating = self._extract_rating(evaluation)

        return Evaluation(
            evaluator=f"{self.name} {self._model.name}",
            query=test_text,
            reference_answer="",
            test_answer="",
            evaluation=evaluation,
            ratings=[rating],
            extras={"criteria": criteria},
        )

    def _extract_rating(self, evaluation: str | None) -> int | None:
        if not evaluation:
            return None
        regex_match = re.search(re.compile(r"\[\[(\d+)\]\]"), evaluation)
        if regex_match:
            rating = int(regex_match.groups()[0])
        else:
            _logger.warning("Could not extract rating from evaluation:%s", evaluation)
            return None
        return rating
