from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from document_agent.agent.nodes import (
    generate,
    reset_turn,
    respond_direct,
    retrieve,
    rewrite_query,
    route,
)
from document_agent.agent.state import AgentState


def route_decision(state: AgentState):
    return state.get("route", "knowledge")


def retrieve_or_generate(state: AgentState):
    if state["retrieved"]:
        return "generate"
    if state.get("attempts", 0) < 2:
        return "rewrite_query"
    return "generate"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)

    builder.add_node("reset_turn", reset_turn)
    builder.add_node("route", route)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("respond_direct", respond_direct)
    builder.add_node("rewrite_query", rewrite_query)

    builder.add_edge(START, "reset_turn")
    builder.add_edge("reset_turn", "route")
    builder.add_conditional_edges("route", route_decision, {
        "knowledge": "retrieve",
        "chitchat": "respond_direct",
    })
    builder.add_conditional_edges("retrieve", retrieve_or_generate)
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("respond_direct", END)

    return builder.compile(checkpointer=checkpointer)
