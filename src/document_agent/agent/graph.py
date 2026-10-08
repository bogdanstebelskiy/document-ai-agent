from langgraph.graph import END, START, StateGraph

from document_agent.agent.nodes import generate, grade, retrieve, rewrite_query
from document_agent.agent.state import AgentState


def route_fn(state: AgentState):
    if state["retrieved"]:
        return "generate"
    if state.get("attempts", 0) < 2:
        return "rewrite_query"
    return "generate"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("grade", grade)
    builder.add_node("rewrite_query", rewrite_query)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_fn)
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
