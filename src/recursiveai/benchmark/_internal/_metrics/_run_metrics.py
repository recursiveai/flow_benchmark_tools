from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field, computed_field

from ._benchmark_metrics import BenchmarkMetrics


class RunMetrics(BaseModel):
    benchmark_metrics: list[BenchmarkMetrics] = Field(exclude=True)

    @computed_field
    @cached_property
    def ratings(self) -> list[float]:
        return [bm_m.mean_rating for bm_m in self.benchmark_metrics]

    @computed_field
    @property
    def sorted_ratings(self) -> list[float]:
        return sorted(self.ratings)

    @computed_field
    @property
    def mean_rating(self) -> float:
        if not self.ratings:
            return 0.0
        return np.mean(self.ratings)

    @computed_field
    @property
    def std_dev(self) -> float:
        if not self.ratings:
            return 0.0
        return np.std(self.ratings)
