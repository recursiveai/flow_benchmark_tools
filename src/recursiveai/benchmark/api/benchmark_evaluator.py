from enum import Enum

from recursiveai.benchmark._internal._evaluators._happy_evaluator import HappyEvaluator


class Evaluator(str, Enum):
    HAPPY = "happy"


def get_evaluator(evaluator: Evaluator):
    match (evaluator):
        case Evaluator.HAPPY:
            return HappyEvaluator()
        case _:
            return HappyEvaluator()
