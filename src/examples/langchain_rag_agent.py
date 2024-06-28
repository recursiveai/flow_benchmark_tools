import asyncio
import logging
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


class LangChainRAGBenchmarkAgent(BenchmarkAgent):
    def __init__(self, open_ai_llm_model: str = "gpt-4o") -> None:
        self._llm_model = ChatOpenAI(model=open_ai_llm_model)

    async def before_run(self, benchmark: Benchmark) -> None:
        docs = set()
        for case in benchmark.cases:
            if "documents" in case.extras:
                docs.update(case.extras["documents"])

        self._doc_chunks = {}
        for doc in docs:
            _, extension = os.path.splitext(doc)
            match (extension):
                case ".pdf":
                    loader = PyPDFLoader(f"data/files/{doc}")
                    self._doc_chunks[doc] = loader.load_and_split()
                case ".docx":
                    loader = Docx2txtLoader(f"data/files/{doc}")
                    self._doc_chunks[doc] = loader.load_and_split()
                case _:
                    _logger.error(
                        "Cannot process document %s. Only PDF and DOCX are supported.",
                        doc,
                    )

    async def run_benchmark_case(self, case: BenchmarkCase) -> BenchmarkCaseResponse:
        if "documents" in case.extras:
            documents = case.extras["documents"]
        else:
            _logger.error("No documents were specified for benchmark case: %s", case)
            return BenchmarkCaseResponse(response=None, exit_code=ExitCode.FAILED)

        chunks = []
        for doc in documents:
            chunks += self._doc_chunks[doc]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        chunks = text_splitter.split_documents(chunks)

        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=OpenAIEmbeddings()
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

        question_answer_chain = create_stuff_documents_chain(self._llm_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        result = rag_chain.invoke({"input": case.query})
        return BenchmarkCaseResponse(
            response=result["answer"], exit_code=ExitCode.SUCCESS
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run = create_run_from_jsonl(
        agent=LangChainRAGBenchmarkAgent(),
        jsonl_file="data/rag_benchmark.jsonl",
    )

    runner = BenchmarkRunner(
        runs=run,
        evaluator=Evaluator.LLM_JUDGE_GEMINI_1_5_PRO,
        results_folder="data/results",
        repeats=1,
    )

    asyncio.run(runner.run())
