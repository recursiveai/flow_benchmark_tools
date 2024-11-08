# Copyright 2024 Recursive AI
import re

from .._benchmark_evaluator import BenchmarkEvaluator
from .._evaluation import Evaluation


class RegexMatchEvaluator(BenchmarkEvaluator):

    async def evaluate(
        self, query: str, reference_answer: str, test_answer: str
    ) -> Evaluation:
        rating = 1
        if re.search(reference_answer, test_answer):
            rating = 10

        return Evaluation(
            evaluator=self.name,
            query=query,
            reference_answer=reference_answer,
            test_answer=test_answer,
            evaluation="",
            ratings=[rating],
        )
