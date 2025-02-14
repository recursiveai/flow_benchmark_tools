# Copyright 2024 Recursive AI

from typing import Any

from pydantic import BaseModel, model_validator

from .exit_code import ExitCode


class BenchmarkCase(BaseModel):
    """
    Data class used to define semantic benchmarks
    """

    query: str
    reference_answer: str | None = None
    reference_answer_file: str | None = None
    labels: list[str] | None = None
    extras: dict[str, Any] | None = None

    @model_validator(mode="after")
    def answer_validator(self):
        if self.labels and "criteria" in self.labels:
            return self
        else:
            if self.reference_answer is None and self.reference_answer_file is None:
                raise ValueError(
                    "Both reference_answer and reference_answer_file are None"
                )
            if self.reference_answer is None and self.reference_answer_file is not None:
                with open(self.reference_answer_file, "r") as file:
                    self.reference_answer = file.read()
        return self


class BenchmarkCaseResponse(BaseModel):
    response: str | None = None
    extras: dict[str, Any] | None = None
    exit_code: ExitCode = ExitCode.SUCCESS
