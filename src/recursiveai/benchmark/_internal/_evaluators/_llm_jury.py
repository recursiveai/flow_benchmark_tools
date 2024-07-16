# Copyright 2024 Recursive AI

import asyncio

from .._benchmark_evaluator import BenchmarkEvaluator
from .._evaluation import Evaluation
from .._llm._llm_model import LLMModel
from ._llm_judge import LLMJudgeEvaluator


class LLMJuryEvaluator(BenchmarkEvaluator):

    def __init__(self, judge_models: list[LLMModel]) -> None:
        super().__init__()
        self._judges = [LLMJudgeEvaluator(model=model) for model in judge_models]

    @property
    def llm_models(self) -> str:
        return ",".join([judge.llm_model for judge in self._judges])

    async def evaluate(
        self, query: str, reference_answer: str, test_answer: str
    ) -> Evaluation:
        evals = await asyncio.gather(
            *[
                judge.evaluate(
                    query=query,
                    reference_answer=reference_answer,
                    test_answer=test_answer,
                )
                for judge in self._judges
            ]
        )

        return Evaluation(
            evaluator=f"{self.name} {self.llm_models}",
            query=query,
            reference_answer=reference_answer,
            test_answer=test_answer,
            evaluation=[evl.evaluation for evl in evals],
            ratings=[evl.rating for evl in evals],
        )
