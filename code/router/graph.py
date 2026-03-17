"""LangGraph StateGraph wiring the routing pipeline."""

from langgraph.graph import StateGraph, START, END
from router.state import RouterState
from router.nodes import (
    scrape_metrics,
    update_cache_directory,
    score_and_route,
    forward_request,
    record_results,
)


def build_graph():
    """Build and compile the routing StateGraph."""
    builder = StateGraph(RouterState)

    # Add nodes
    builder.add_node("scrape_metrics", scrape_metrics)
    builder.add_node("update_cache_directory", update_cache_directory)
    builder.add_node("score_and_route", score_and_route)
    builder.add_node("forward_request", forward_request)
    builder.add_node("record_results", record_results)

    # Wire edges: linear pipeline
    builder.add_edge(START, "scrape_metrics")
    builder.add_edge("scrape_metrics", "update_cache_directory")
    builder.add_edge("update_cache_directory", "score_and_route")
    builder.add_edge("score_and_route", "forward_request")
    builder.add_edge("forward_request", "record_results")
    builder.add_edge("record_results", END)

    return builder.compile()


# Compiled graph — import this from server.py
routing_graph = build_graph()
