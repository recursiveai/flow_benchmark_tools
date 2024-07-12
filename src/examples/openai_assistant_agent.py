# Copyright 2024 Recursive AI

import asyncio
import logging

from openai import AsyncClient
from openai.types import FileObject
from openai.types.beta.threads.message import Message

from recursiveai.benchmark.api import (
    Benchmark,
    BenchmarkAgent,
    BenchmarkCase,
    BenchmarkCaseResponse,
    BenchmarkRunner,
    Evaluator,
    ExitCode,
)
from recursiveai.benchmark.api.util import create_run_from_jsonl

_logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks.\n"
    "If you don't know the answer, say that you don't know.\n"
    "Use three sentences maximum and keep the answer concise.\n"
)


class OpenAIAssistantBenchmarkAgent(BenchmarkAgent):
    def __init__(
        self,
        model: str = "gpt-4o",
    ) -> None:
        self._client = AsyncClient(timeout=120.0)
        self._model = model

    async def before_run(self, _: Benchmark) -> None:
        self._assistant = await self._client.beta.assistants.create(
            name="Benchmark Agent",
            instructions=_SYSTEM_PROMPT,
            model=self._model,
            temperature=0.0,
            tools=[{"type": "file_search"}],
        )

    async def after_run(self, _: Benchmark) -> None:
        await self._client.beta.assistants.delete(self._assistant.id)

    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        if "documents" in case.extras:
            documents: list[str] = case.extras["documents"]
        else:
            _logger.error("No documents were specified for benchmark case: %s", case)
            return BenchmarkCaseResponse(response=None, exit_code=ExitCode.FAILED)

        try:
            files: list[FileObject] = []
            file_map: dict[str, str] = {}
            for doc in documents:
                with open(f"data/files/{doc}", "rb") as base_file:
                    file = await self._client.files.create(
                        file=base_file, purpose="assistants"
                    )

                if file:
                    file_map[file.id] = doc
                    files.append(file)

            vector_store = await self._client.beta.vector_stores.create(
                name="benchmark_files",
                expires_after={"anchor": "last_active_at", "days": 1},
            )

            await self._client.beta.vector_stores.file_batches.create_and_poll(
                vector_store_id=vector_store.id,
                file_ids=[file.id for file in files],
            )

            thread = await self._client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": case.query,
                    }
                ],
                tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
            )

            run = await self._client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=self._assistant.id,
            )

            messages: list[Message] = [
                message
                async for message in await self._client.beta.threads.messages.list(
                    thread_id=thread.id,
                    run_id=run.id,
                )
            ]
            message_content = messages[0].content[0].text.value

            annotations = messages[0].content[0].text.annotations
            for annotation in annotations:
                message_content = message_content.replace(annotation.text, "")

        finally:
            for file in files:
                await self._client.files.delete(file.id)

            resources = thread.tool_resources
            if resources.file_search and resources.file_search.vector_store_ids:
                for vs_id in resources.file_search.vector_store_ids:
                    await self._client.beta.vector_stores.delete(vs_id)
            await self._client.beta.threads.delete(thread.id)

        return BenchmarkCaseResponse(
            response=message_content, exit_code=ExitCode.SUCCESS
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run = create_run_from_jsonl(
        agent=OpenAIAssistantBenchmarkAgent(),
        jsonl_file="data/rag_benchmark.jsonl",
    )

    runner = BenchmarkRunner(
        runs=run,
        evaluator=Evaluator.LLM_JURY_GPT_CLAUDE_GEMINI_LOW,
        results_folder="data/results",
        repeats=1,
        parallel=True,
    )

    asyncio.run(runner.run())
