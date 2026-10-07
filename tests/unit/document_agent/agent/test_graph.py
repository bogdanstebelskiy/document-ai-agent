from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from document_agent.agent.graph import build_graph
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


def _fake_store(chunks):
    store = MagicMock()
    store.search.return_value = [(c, 0.9) for c in chunks]
    return store


def _fake_llm(answer_text):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=answer_text)
    return llm


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_graph_produces_answer_and_citations(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        "Python was created by Guido van Rossum [1] and version 3.13 came out in 2024 [2]."
    )

    graph = build_graph()
    result = graph.invoke({"question": "Who created Python?", "messages": []})

    assert "answer" in result
    assert "citations" in result
    assert "retrieved" in result
    assert len(result["retrieved"]) == 2
    assert result["citations"] == ["[1] notes.md (chunk 0)", "[2] faq.md (chunk 0)"]
    assert "[1]" in result["answer"]


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_graph_citations_only_include_referenced_sources(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        "Python was created by Guido van Rossum [1]."
    )

    graph = build_graph()
    result = graph.invoke({"question": "Who created Python?", "messages": []})

    assert result["citations"] == ["[1] notes.md (chunk 0)"]


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_graph_shows_each_chunk_separately_from_same_file(mock_get_llm, mock_vector_cls):
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
    mock_vector_cls.default.return_value = _fake_store(same_file_chunks)
    mock_get_llm.return_value = _fake_llm(
        "Both parts say the same thing [1] [2]."
    )

    graph = build_graph()
    result = graph.invoke({"question": "What do the notes say?", "messages": []})

    assert result["citations"] == [
        "[1] notes.md (chunk 0)",
        "[2] notes.md (chunk 1)",
    ]


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_graph_no_citations_when_none_referenced(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        "I don't have enough information to answer that."
    )

    graph = build_graph()
    result = graph.invoke({"question": "What is quantum physics?", "messages": []})

    assert result["citations"] == []
