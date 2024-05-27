import logging
import re

from .._benchmark_evaluator import BenchmarkEvaluator
from .._evaluation import Evaluation
from .._llm._llm_model import ChatMessage, LLMModel

_logger = logging.getLogger(__name__)

_REFERENCED_JUDGE_SYSTEM_PROMPT = (
    "You will be given two answers to a user question: a reference answer and a test answer.\n"
    "Assume that the reference answer is the perfect answer to the user question.\n"
    "Please act as an impartial and objective judge and evaluate the quality of the test answer by comparing it with the reference answer.\n"
    "Begin your evaluation by providing a short description of the similarities and dissimilarities between the two answers, including any information missing from the test answer.\n"
    "You should compare only the information that is relevant to the user question.\n"
    "Do not allow the length or the format of the answers to influence your evaluation.\n"
    'After providing your short description, you must rate the test answer on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"\n'
)

_REFERENCED_JUDGE_USER_PROMPT = (
    "[User question]\n"
    "{question}\n"
    "\n"
    "[Reference answer]\n"
    "{reference_answer}\n"
    "\n"
    "[Test answer]\n"
    "{test_answer}\n"
)


class LLMJudgeEvaluator(BenchmarkEvaluator):

    def __init__(self, model: LLMModel) -> None:
        super().__init__()
        self._model = model

    async def evaluate(
        self, query: str, reference_answer: str, test_answer: str
    ) -> Evaluation:
        user_prompt = _REFERENCED_JUDGE_USER_PROMPT.format(
            question=query,
            reference_answer=reference_answer,
            test_answer=test_answer,
        )

        chat = [
            ChatMessage(content=_REFERENCED_JUDGE_SYSTEM_PROMPT, role="system"),
            ChatMessage(content=user_prompt, role="user"),
        ]

        evaluation = await self._model.async_chat_completion(
            chat,
            temperature=0,
            max_tokens=1024,
            timeout=60,
        )

        rating = self._extract_rating(evaluation)

        return Evaluation(
            evaluator=f"{self.name} {self._model.name}",
            query=query,
            reference_answer=reference_answer,
            test_answer=test_answer,
            evaluation=evaluation,
            rating=rating,
        )

    def _extract_rating(self, evaluation: str | None) -> int | None:
        if not evaluation:
            return None
        regex_match = re.search(re.compile(r"\[\[(\d+)\]\]"), evaluation)
        if regex_match:
            rating = int(regex_match.groups()[0])
        else:
            _logger.warning(f"Could not extract rating from evaluation:{evaluation}")
            return None
        return rating