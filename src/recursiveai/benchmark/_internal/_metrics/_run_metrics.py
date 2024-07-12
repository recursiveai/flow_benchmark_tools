# Copyright 2024 Recursive AI

from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field, computed_field

from ._benchmark_metrics import BenchmarkMetrics


class RunMetrics(BaseModel):
    benchmark_metrics: list[BenchmarkMetrics] = Field(exclude=True)

    @computed_field
    @property
    def num_benchmarks(self) -> int:
        return len(self.benchmark_metrics)

    @computed_field
    @cached_property
    def ratings(self) -> list[float | None]:
        return [bm_m.mean_rating for bm_m in self.benchmark_metrics]

    @computed_field
    @property
    def num_valid_ratings(self) -> int:
        return len(self.valid_ratings)

    @computed_field
    @cached_property
    def valid_ratings(self) -> list[float]:
        return list(filter(None, self.ratings))

    @computed_field
    @property
    def sorted_ratings(self) -> list[float]:
        return [value for _, value in self.sorted_enumerated_ratings]

    @computed_field
    @cached_property
    def sorted_enumerated_ratings(self) -> list[tuple[int, float]]:
        return sorted(enumerate(self.ratings), key=lambda x: (x[1] is not None, x[1]))

    @computed_field
    @property
    def histogram(self) -> dict[str, int]:
        bins = [-0.5, 4.5, 7.5, 10.5]
        counts, _ = np.histogram(self.valid_ratings, bins=bins)
        if len(counts) == 3:
            return {
                "invalid": len(self.ratings) - len(self.valid_ratings),
                "poor": int(counts[0]),
                "fair": int(counts[1]),
                "good": int(counts[2]),
            }
        return {}

    @computed_field
    @property
    def mean_rating(self) -> float | None:
        if not self.valid_ratings:
            return None
        return np.mean(self.valid_ratings)

    @computed_field
    @property
    def std_dev(self) -> float | None:
        if not self.valid_ratings:
            return None
        return np.std(self.valid_ratings)
