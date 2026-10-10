from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from document_agent.domain import Chunk


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    search_query: str
    retrieved: list[Chunk]
    attempts: int
    route: str
    answer: str
    citations: list[str]


class RouteDecision(BaseModel):
    route: str