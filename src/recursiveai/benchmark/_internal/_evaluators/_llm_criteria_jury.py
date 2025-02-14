# Copyright 2024 Recursive AI

import asyncio

from recursiveai.benchmark._internal._evaluators._llm_criteria_judge import (
    LLMCriteriaJudgeEvaluator,
)

from .._criteria_evaluator import CriteriaEvaluator
from .._evaluation import Evaluation
from .._llm._llm_model import LLMModel


class LLMCriteriaJuryEvaluator(CriteriaEvaluator):

    def __init__(self, judge_models: list[LLMModel]) -> None:
        super().__init__()
        self._judges = [
            LLMCriteriaJudgeEvaluator(model=model) for model in judge_models
        ]

    @property
    def llm_models(self) -> str:
        return ",".join([judge.llm_model for judge in self._judges])

    async def evaluate(self, criteria: str, test_text: str) -> Evaluation:
        evals = await asyncio.gather(
            *[
                judge.evaluate(
                    criteria=criteria,
                    test_text=test_text,
                )
                for judge in self._judges
            ]
        )

        return Evaluation(
            evaluator=f"{self.name} {self.llm_models}",
            query=test_text,
            reference_answer="",
            test_answer="",
            evaluation=[evl.evaluation for evl in evals],
            ratings=[evl.rating for evl in evals],
            extras={"criteria": criteria},
        )
