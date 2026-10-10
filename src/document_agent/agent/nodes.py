import logging
import re
from pathlib import PurePath

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
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
        question=state["search_query"],
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
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer_text),
        ],
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
                f"Does this chunk contain ANY information that could help "
                f"answer the question '{state['search_query']}'? "
                f"When in doubt, mark as relevant.\n\n{chunk.text}"
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

    recent_messages = state.get("messages", [])[-6:]
    if recent_messages:
        history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
        prompt = (
            f"Given this conversation history:\n{history}\n\n"
            f"Rephrase this question into a standalone search query "
            f"that resolves any pronouns or references: {state['question']}"
        )
    else:
        prompt = f"Rephrase this question for better search: {state['question']}"

    updated_query = get_llm().invoke(prompt).content
    updated_attempts = state.get("attempts", 0) + 1

    return {
        "attempts": updated_attempts,
        "search_query": updated_query,
    }


def reset_turn(state: AgentState) -> dict:
    question = state["question"]
    recent_messages = state.get("messages", [])[-6:]
    if recent_messages:
        history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
        search_query = get_llm().invoke(
            "Given this conversation history:\n"
            f"{history}\n\n"
            "The user now asks a follow-up question. Rewrite it as a "
            "standalone search query by replacing pronouns (it, this, "
            "that, they, etc.) with the specific subject from the "
            "conversation. The subject is usually the main topic being "
            "discussed, not a sub-concept.\n\n"
            "Examples:\n"
            "- History about PostgreSQL connection pooling, follow-up "
            "\"Does it support transactions?\" -> "
            "\"Does PostgreSQL support transactions?\"\n"
            "- History about React hooks, follow-up "
            "\"How do I test them?\" -> \"How do I test React hooks?\"\n\n"
            f"Follow-up: {question}\n"
            "Standalone query:"
        ).content.strip()
    else:
        search_query = question

    return {
        "retrieved": [],
        "search_query": search_query,
        "attempts": 0,
        "answer": "",
        "citations": [],
    }
