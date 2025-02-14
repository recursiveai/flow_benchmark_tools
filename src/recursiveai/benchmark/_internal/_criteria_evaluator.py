# Copyright 2024 Recursive AI

from abc import ABC, abstractmethod

from ._evaluation import Evaluation


class CriteriaEvaluator(ABC):

    @abstractmethod
    async def evaluate(self, criteria: str, test_text: str) -> Evaluation:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__
