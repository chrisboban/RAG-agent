from langgraph.graph import StateGraph, START, END

from nodes.router import router
from nodes.retriever import retriever
from nodes.generator import generator
from state import State


def decide_route(state):
    return state.get("route", "answer")


def build_graph():
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("router", router)
    builder.add_node("retriever", retriever)
    builder.add_node("generator", generator)

    # Entry point
    builder.add_edge(START,"router")

    # Conditional routing
    builder.add_conditional_edges(
        "router",
        decide_route,
        {
            "retrieve": "retriever",
            "answer": "generator"
        }
    )

    # Flow after decision is mades
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile()
