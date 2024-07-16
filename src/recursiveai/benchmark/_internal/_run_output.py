# Copyright 2024 Recursive AI

from pydantic import BaseModel, computed_field

from ._benchmark_output import BenchmarkOutput
from ._metrics._run_metrics import RunMetrics


class RunOutput(BaseModel):
    date: str
    agent_name: str
    benchmark_outputs: list[BenchmarkOutput]

    @computed_field
    @property
    def metrics(self) -> RunMetrics:
        benchmark_metrics = [bm.metrics for bm in self.benchmark_outputs]
        return RunMetrics(benchmark_metrics=benchmark_metrics)
