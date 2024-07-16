# Copyright 2024 Recursive AI

import asyncio
import logging
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

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
    "Use the following pieces of retrieved context to answer the question.\n"
    "If you don't know the answer, say that you don't know.\n"
    "Use three sentences maximum and keep the answer concise.\n"
    "\n"
    "{context}"
)


class _VectorStoreData(BaseModel):
    texts: list[str]
    metadata: list[dict]
    embeddings: list[list[float]]


class LangChainRAGBenchmarkAgent(BenchmarkAgent):
    def __init__(self, open_ai_llm_model: str = "gpt-4o") -> None:
        self._open_ai_llm_model = open_ai_llm_model
        self._embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    async def before_run(self, benchmark: Benchmark) -> None:
        docs = set()
        for case in benchmark.cases:
            if "documents" in case.extras:
                docs.update(case.extras["documents"])

        vs_data = await asyncio.gather(*[self._load_document(doc) for doc in docs])
        self._vs_data: dict[str, _VectorStoreData] = {}
        for idx, doc in enumerate(docs):
            self._vs_data[doc] = vs_data[idx]

    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        if "documents" in case.extras:
            documents = case.extras["documents"]
        else:
            return BenchmarkCaseResponse(response=None, exit_code=ExitCode.FAILED)

        texts = []
        metadata = []
        embeddings = []
        for doc in documents:
            texts += self._vs_data[doc].texts
            metadata += self._vs_data[doc].metadata
            embeddings += self._vs_data[doc].embeddings

        vectorstore = await FAISS.afrom_embeddings(
            text_embeddings=zip(texts, embeddings),
            embedding=self._embedder,
            metadatas=metadata,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        llm_model = ChatOpenAI(model=self._open_ai_llm_model)
        question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        result = await rag_chain.ainvoke({"input": case.query})
        return BenchmarkCaseResponse(
            response=result["answer"], exit_code=ExitCode.SUCCESS
        )

    async def _load_document(self, filename: str) -> _VectorStoreData:
        _, extension = os.path.splitext(filename)
        match (extension):
            case ".pdf":
                loader = PyPDFLoader(f"data/files/{filename}")
                chunks = await loader.aload()
            case ".docx":
                loader = Docx2txtLoader(f"data/files/{filename}")
                chunks = await loader.aload()
            case _:
                chunks = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        chunks = text_splitter.split_documents(chunks)

        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        embeddings = await self._embedder.aembed_documents(texts=texts)

        return _VectorStoreData(texts=texts, metadata=metadata, embeddings=embeddings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run = create_run_from_jsonl(
        agent=LangChainRAGBenchmarkAgent(),
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
