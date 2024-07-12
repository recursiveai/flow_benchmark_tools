# Copyright 2024 Recursive AI

from abc import ABC, abstractmethod

from ._evaluation import Evaluation


class BenchmarkEvaluator(ABC):

    @abstractmethod
    async def evaluate(
        self, query: str, reference_answer: str, test_answer: str
    ) -> Evaluation:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__
