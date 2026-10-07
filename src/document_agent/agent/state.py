from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from document_agent.domain import Chunk


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    query: str
    retrieved: list[Chunk]
    relevant: bool
    attempts: int
    answer: str
    citations: list[str]