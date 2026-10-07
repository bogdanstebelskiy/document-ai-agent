from langgraph.graph import END, START, StateGraph

from document_agent.agent.nodes import generate, retrieve
from document_agent.agent.state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
