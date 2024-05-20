from pydantic import BaseModel, computed_field

from ..api.benchmark import Benchmark
from ._evaluation import Evaluation


class BenchmarkOutput(BaseModel):
    info: Benchmark
    repeats: int = 1
    evaluations: list[Evaluation | None]

    @computed_field
    @property
    def average_rating(self) -> float:
        valid_evals = list(filter(None, self.evaluations))
        ratings = [eval.rating for eval in valid_evals if eval.rating is not None]
        if not ratings:
            return 0.0
        return sum(ratings) / len(ratings)
