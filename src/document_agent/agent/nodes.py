import re
from pathlib import PurePath

from document_agent.agent.prompts import GENERATION_PROMPT
from document_agent.agent.state import AgentState
from document_agent.models import get_llm
from document_agent.store.vector import VectorStore


def _source_label(chunk) -> str:
    raw = chunk.metadata.get("source_uri", "unknown")
    return PurePath(raw).name


def retrieve(state: AgentState) -> dict:
    results = VectorStore.default().search(state["question"], k=5)
    seen = set()
    chunks = []
    for chunk, _score in results:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return {"retrieved": chunks}


def generate(state: AgentState) -> dict:
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
