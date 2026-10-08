from unittest.mock import MagicMock, patch

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from document_agent.agent.graph import build_graph
from document_agent.agent.state import Grade
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


def _fake_llm(invoke_results, grade_results=None):
    llm = MagicMock()
    if isinstance(invoke_results, str):
        llm.invoke.return_value = AIMessage(content=invoke_results)
    else:
        llm.invoke.side_effect = [AIMessage(content=t) for t in invoke_results]

    grader = MagicMock()
    if grade_results is None:
        grader.invoke.return_value = Grade(relevant=True)
    elif isinstance(grade_results, Exception):
        grader.invoke.side_effect = grade_results
    elif isinstance(grade_results, list):
        grader.invoke.side_effect = grade_results
    else:
        grader.invoke.return_value = grade_results
    llm.with_structured_output.return_value = grader

    return llm


def _invoke(question):
    graph = build_graph()
    return graph.invoke({
        "question": question,
        "search_query": question,
        "messages": [],
    })


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_graph_produces_answer_and_citations(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        "Python was created by Guido van Rossum [1] and version 3.13 came out in 2024 [2]."
    )

    result = _invoke("Who created Python?")

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

    result = _invoke("Who created Python?")

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

    result = _invoke("What do the notes say?")

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

    result = _invoke("What is quantum physics?")

    assert result["citations"] == []


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_relevant_chunks_go_straight_to_generate(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        "Python was created by Guido van Rossum [1].",
        grade_results=Grade(relevant=True),
    )

    result = _invoke("Who created Python?")

    assert len(result["retrieved"]) == 2
    assert result.get("attempts", 0) == 0


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_retry_on_irrelevant_then_succeed(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        invoke_results=["better search query", "The answer is 42 [1]."],
        grade_results=[
            Grade(relevant=False),
            Grade(relevant=False),
            Grade(relevant=True),
            Grade(relevant=True),
        ],
    )

    result = _invoke("vague question")

    assert result["attempts"] == 1
    assert len(result["retrieved"]) == 2
    assert "answer" in result


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_exhausts_retries_then_generates(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        invoke_results=["rewrite 1", "rewrite 2", "Not found in your notes."],
        grade_results=Grade(relevant=False),
    )

    result = _invoke("something completely unrelated")

    assert result["attempts"] == 2
    assert result["retrieved"] == []
    assert "answer" in result


@patch("document_agent.agent.nodes.VectorStore")
@patch("document_agent.agent.nodes.get_llm")
def test_parse_failure_keeps_chunk_as_relevant(mock_get_llm, mock_vector_cls):
    chunks = _fake_chunks()
    mock_vector_cls.default.return_value = _fake_store(chunks)
    mock_get_llm.return_value = _fake_llm(
        "Python was created by Guido van Rossum [1].",
        grade_results=OutputParserException("bad output"),
    )

    result = _invoke("Who created Python?")

    assert len(result["retrieved"]) == 2
    assert "answer" in result
