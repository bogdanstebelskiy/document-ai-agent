from unittest.mock import MagicMock

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from document_agent.agent.graph import build_graph
from document_agent.agent.nodes import MAX_DISTANCE
from document_agent.agent.state import AgentDeps, RouteDecision
from document_agent.domain import Chunk


def _fake_chunks():
    return [
        Chunk(
            chunk_id="notes:0",
            doc_id="notes",
            index=0,
            text="Python was created by Guido van Rossum.",
            metadata={"doc_id": "notes", "index": 0, "source_uri": "/docs/notes.md"},
        ),
        Chunk(
            chunk_id="faq:0",
            doc_id="faq",
            index=0,
            text="Python 3.13 was released in October 2024.",
            metadata={"doc_id": "faq", "index": 0, "source_uri": "/docs/faq.md"},
        ),
    ]


def _fake_store(chunks, score=0.5):
    store = MagicMock()
    if callable(score):
        store.search.side_effect = score
    else:
        store.search.return_value = [(c, score) for c in chunks]
    return store


def _fake_llm(invoke_results, route_to="knowledge"):
    llm = MagicMock()
    if isinstance(invoke_results, str):
        llm.invoke.return_value = AIMessage(content=invoke_results)
    else:
        llm.invoke.side_effect = [AIMessage(content=t) for t in invoke_results]

    router = MagicMock()
    router.invoke.return_value = RouteDecision(route=route_to)
    llm.with_structured_output.return_value = router

    return llm


def _invoke(question, deps):
    graph = build_graph(deps=deps)
    return graph.invoke({
        "question": question,
        "search_query": question,
        "messages": [],
    })


def test_graph_produces_answer_and_citations():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Python was created by Guido van Rossum [1] and version 3.13 came out in 2024 [2]."),
        vector_store=_fake_store(chunks),
    )

    result = _invoke("Who created Python?", deps)

    assert "answer" in result
    assert "citations" in result
    assert "retrieved" in result
    assert len(result["retrieved"]) == 2
    assert result["citations"] == ["[1] notes.md (chunk 0)", "[2] faq.md (chunk 0)"]
    assert "[1]" in result["answer"]


def test_graph_citations_only_include_referenced_sources():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Python was created by Guido van Rossum [1]."),
        vector_store=_fake_store(chunks),
    )

    result = _invoke("Who created Python?", deps)

    assert result["citations"] == ["[1] notes.md (chunk 0)"]


def test_graph_shows_each_chunk_separately_from_same_file():
    same_file_chunks = [
        Chunk(
            chunk_id="notes:0", doc_id="notes", index=0,
            text="First part of the notes.",
            metadata={"doc_id": "notes", "index": 0, "source_uri": "/docs/notes.md"},
        ),
        Chunk(
            chunk_id="notes:1", doc_id="notes", index=1,
            text="Second part of the notes.",
            metadata={"doc_id": "notes", "index": 1, "source_uri": "/docs/notes.md"},
        ),
    ]
    deps = AgentDeps(
        llm=_fake_llm("Both parts say the same thing [1] [2]."),
        vector_store=_fake_store(same_file_chunks),
    )

    result = _invoke("What do the notes say?", deps)

    assert result["citations"] == [
        "[1] notes.md (chunk 0)",
        "[2] notes.md (chunk 1)",
    ]


def test_graph_no_citations_when_none_referenced():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("I don't have enough information to answer that."),
        vector_store=_fake_store(chunks),
    )

    result = _invoke("What is quantum physics?", deps)

    assert result["citations"] == []


def test_good_scores_go_straight_to_generate():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Python was created by Guido van Rossum [1]."),
        vector_store=_fake_store(chunks, score=0.4),
    )

    result = _invoke("Who created Python?", deps)

    assert len(result["retrieved"]) == 2
    assert result.get("attempts", 0) == 0


def test_bad_scores_trigger_rewrite_then_succeed():
    chunks = _fake_chunks()
    call_count = 0

    def search_with_improving_scores(query, k=5):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [(c, 0.95) for c in chunks]
        return [(c, 0.4) for c in chunks]

    deps = AgentDeps(
        llm=_fake_llm(invoke_results=["better search query", "The answer is 42 [1]."]),
        vector_store=_fake_store(chunks, score=search_with_improving_scores),
    )

    result = _invoke("vague question", deps)

    assert result["attempts"] == 1
    assert len(result["retrieved"]) == 2
    assert "answer" in result


def test_exhausts_retries_then_generates():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm(invoke_results=["rewrite 1", "rewrite 2", "Not found in your notes."]),
        vector_store=_fake_store(chunks, score=0.95),
    )

    result = _invoke("something completely unrelated", deps)

    assert result["attempts"] == 2
    assert result["retrieved"] == []
    assert "answer" in result


def test_score_threshold_filters_chunks():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm(invoke_results=["rewrite 1", "rewrite 2", "Not found."]),
        vector_store=_fake_store(chunks, score=MAX_DISTANCE + 0.1),
    )

    result = _invoke("irrelevant query", deps)

    assert result["retrieved"] == []


def test_chunks_at_threshold_are_kept():
    chunks = _fake_chunks()
    deps = AgentDeps(
        llm=_fake_llm("Found it [1]."),
        vector_store=_fake_store(chunks, score=MAX_DISTANCE),
    )

    result = _invoke("borderline query", deps)

    assert len(result["retrieved"]) == 2


# --- Routing tests ---


def test_chitchat_skips_retrieval():
    store = _fake_store([])
    deps = AgentDeps(
        llm=_fake_llm("You're welcome!", route_to="chitchat"),
        vector_store=store,
    )

    result = _invoke("thanks!", deps)

    assert result["answer"] == "You're welcome!"
    assert result["retrieved"] == []
    assert result["citations"] == []
    store.search.assert_not_called()


def test_knowledge_routes_to_retrieval():
    chunks = _fake_chunks()
    store = _fake_store(chunks, score=MAX_DISTANCE + 0.1)
    deps = AgentDeps(
        llm=_fake_llm("Python was created by Guido van Rossum [1].", route_to="knowledge"),
        vector_store=store,
    )

    result = _invoke("Who created Python?", deps)

    assert result["retrieved"] == []
    store.search.assert_called()


def test_route_parse_failure_defaults_to_knowledge():
    chunks = _fake_chunks()
    llm = _fake_llm("Python was created by Guido van Rossum [1].")
    llm.with_structured_output.return_value.invoke.side_effect = OutputParserException("bad output")
    deps = AgentDeps(
        llm=llm,
        vector_store=_fake_store(chunks),
    )

    result = _invoke("Who created Python?", deps)

    assert len(result["retrieved"]) == 2
    assert "answer" in result
