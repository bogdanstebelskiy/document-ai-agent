import logging
import re
from pathlib import PurePath

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

logger = logging.getLogger(__name__)

from document_agent.agent.prompts import GENERATION_PROMPT
from document_agent.agent.state import AgentState, Grade
from document_agent.models import get_llm
from document_agent.store.vector import VectorStore


def _source_label(chunk) -> str:
    raw = chunk.metadata.get("source_uri", "unknown")
    return PurePath(raw).name


def retrieve(state: AgentState) -> dict:
    query = state["search_query"]
    logger.info("retrieve: query=%r", query)
    results = VectorStore.default().search(query, k=5)
    seen = set()
    chunks = []
    for chunk, _score in results:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return {"retrieved": chunks}


def generate(state: AgentState) -> dict:
    logger.info("generate: %d chunks", len(state["retrieved"]))
    chunks = state["retrieved"]
    context = "\n\n".join(
        f"[{i + 1}] ({_source_label(chunk)})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    prompt = GENERATION_PROMPT.format(
        question=state["question"],
        context=context,
    )

    response = get_llm().invoke(prompt)
    answer_text = response.content

    cited_indices = {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}
    citations = [
        f"[{i}] {_source_label(chunks[i - 1])} (chunk {chunks[i - 1].index})"
        for i in sorted(cited_indices)
        if 1 <= i <= len(chunks)
    ]

    return {
        "answer": answer_text,
        "citations": citations,
    }


def grade(state: AgentState) -> dict:
    logger.info("grade: %d chunks, attempts=%d", len(state["retrieved"]), state.get("attempts", 0))
    grader = get_llm().with_structured_output(Grade)
    relevant_chunks = []

    for chunk in state["retrieved"]:
        try:
            result = grader.invoke(
                f"Is this chunk relevant to the question '{state['question']}'?\n\n{chunk.text}"
            )

            if result.relevant:
                relevant_chunks.append(chunk)
        except (OutputParserException, ValidationError):
            relevant_chunks.append(chunk)

    return {
        "retrieved": relevant_chunks
    }


def rewrite_query(state: AgentState) -> dict:
    logger.info("rewrite_query: attempts=%d, question=%r", state.get("attempts", 0), state["question"])
    updated_query = get_llm().invoke(f"Rephrase this question for better search: {state['question']}").content
    updated_attempts = state.get("attempts", 0) + 1

    return {
        "attempts": updated_attempts,
        "search_query": updated_query
    }
