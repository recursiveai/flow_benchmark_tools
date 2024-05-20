from pydantic import BaseModel

from ._benchmark_output import BenchmarkOutput


class RunOutput(BaseModel):
    date: str
    agent_name: str
    benchmark_outputs: list[BenchmarkOutput]
