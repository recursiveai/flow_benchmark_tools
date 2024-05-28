from typing import Any

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class Evaluation(BaseModel):
    evaluator: str
    query: str
    reference_answer: str
    test_answer: str
    evaluation: str | None
    rating: int | None = None
    rating_min: int = 1
    rating_max: int = 10
    extras: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_rating_within_bounds(self) -> Self:
        if self.rating:
            if self.rating < self.rating_min or self.rating > self.rating_max:
                raise ValueError(
                    f"rating:{self.rating} is out of bounds (rating_min:{self.rating_min} rating_max:{self.rating_max})"
                )
        return self
