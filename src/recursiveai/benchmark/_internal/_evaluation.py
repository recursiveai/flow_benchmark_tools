# Copyright 2024 Recursive AI

from typing import Any

import numpy as np
from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self


class Evaluation(BaseModel):
    evaluator: str
    query: str
    reference_answer: str
    test_answer: str
    evaluation: str | list[str] | None
    ratings: list[int | None]
    rating_min: int = 1
    rating_max: int = 10
    extras: dict[str, Any] | None = None

    @computed_field
    @property
    def rating(self) -> int | None:
        valid_ratings = list(filter(None, self.ratings))
        if not valid_ratings:
            return None
        return round(np.mean(valid_ratings))

    @model_validator(mode="after")
    def check_ratings_within_bounds(self) -> Self:
        for rating in self.ratings:
            if rating:
                if rating < self.rating_min or rating > self.rating_max:
                    raise ValueError(
                        f"rating:{rating} is out of bounds (rating_min:{self.rating_min} rating_max:{self.rating_max})"
                    )
        return self
