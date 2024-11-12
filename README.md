# Flow Benchmark Tools

Create and run LLM benchmarks.

## Installation

Just the library:
```sh
pip install flow-benchmark-tools:1.1.0
```

Library + Example benchmarks (see below):
```sh
pip install "flow-benchmark-tools[examples]:1.1.0"
```

## Usage

* Create an agent by inheriting [BenchmarkAgent](src/recursiveai/benchmark/api/benchmark_agent.py) and implementing the `run_benchmark_case` method.

* Create a [Benchmark](src/recursiveai/benchmark/api/benchmark.py) by compiling a list of [BenchmarkCases](src/recursiveai/benchmark/api/benchmark_case.py). These can be read from a JSONL file.

* Associate agent and benchmark in a [BenchmarkRun](src/recursiveai/benchmark/api/benchmark_run.py).

* Use a [BenchmarkRunner](src/recursiveai/benchmark/api/benchmark_runner.py) to run your BenchmarkRun.

## Running example benchmarks

Two end-to-end benchmark examples are provided in the [examples](src/examples) folder: a [LangChain RAG](src/examples/langchain_rag_agent.py) application and an [OpenAI Assistant](src/examples/openai_assistant_agent.py) agent.

To run the LangChain RAG benchmark:
```sh
python src/examples/langchain_rag_agent.py
```

To run the OpenAI Assistant benchmark:
```sh
python src/examples/openai_assistant_agent.py
```

The benchmark cases are defined in [data/rag_benchmark.jsonl](data/rag_benchmark.jsonl).

The two examples follow the typical usage pattern of the library:
* define an agent by implementing the [BenchmarkAgent](src/recursiveai/benchmark/api/benchmark_agent.py) interface and overriding the `run_benchmark_case` method (you can also override the `before` and `after` methods, if needed),
* create a set of benchmark cases, typically as a JSONL file such as [data/rag_benchmark.jsonl](data/rag_benchmark.jsonl),
* use a [BenchmarkRunner](src/recursiveai/benchmark/api/benchmark_runner.py) to run the benchmark.
