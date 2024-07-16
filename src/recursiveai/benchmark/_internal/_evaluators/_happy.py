# Copyright 2024 Recursive AI

from .._benchmark_evaluator import BenchmarkEvaluator
from .._evaluation import Evaluation


class HappyEvaluator(BenchmarkEvaluator):

    def __init__(self, rating: int = 10) -> None:
        super().__init__()
        self._rating = rating

    async def evaluate(
        self, query: str, reference_answer: str, test_answer: str
    ) -> Evaluation:
        return Evaluation(
            evaluator=self.name,
            query=query,
            reference_answer=reference_answer,
            test_answer=test_answer,
            evaluation="",
            ratings=[self._rating],
        )
