# Copyright 2024 Recursive AI

from pydantic import BaseModel, computed_field

from ..api.benchmark_case import BenchmarkCase
from ._evaluation import Evaluation
from ._metrics._benchmark_metrics import BenchmarkMetrics


class BenchmarkOutput(BaseModel):
    id: int
    info: BenchmarkCase
    repeats: int = 1
    evaluations: list[Evaluation | None]
    mean_case_runtime: float | None = None
    total_runtime: float | None = None

    @computed_field
    @property
    def metrics(self) -> BenchmarkMetrics:
        return BenchmarkMetrics(evals=self.evaluations)
