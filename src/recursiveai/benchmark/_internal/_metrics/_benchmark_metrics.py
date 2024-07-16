# Copyright 2024 Recursive AI

from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field, computed_field

from .._evaluation import Evaluation


class BenchmarkMetrics(BaseModel):
    evals: list[Evaluation | None] = Field(exclude=True)

    @computed_field
    @property
    def num_evals(self) -> int:
        return len(self.evals)

    @computed_field
    @property
    def num_valid_ratings(self) -> int:
        return len(self.valid_ratings)

    @computed_field
    @cached_property
    def valid_ratings(self) -> list[int]:
        valid_evals = list(filter(None, self.evals))
        return [eval.rating for eval in valid_evals if eval.rating is not None]

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
