"""Shared state that flows through the LangGraph routing graph."""

from typing import Any
from typing_extensions import TypedDict


class InstanceState(TypedDict):
    """Metrics snapshot for a single vLLM instance."""
    queue_depth: int
    num_running: int
    cache_hit_rate: float
    avg_ttft: float


class RouterState(TypedDict):
    """State passed between LangGraph nodes on each request."""

    # Input
    request: dict[str, Any]

    # Populated by scrape_metrics
    instance_states: list[InstanceState]

    # Populated by update_cache_directory
    cache_affinity: list[float]

    # Populated by score_and_route
    chosen_instance: int

    # Populated by forward_request
    response: dict[str, Any]

    # Populated by record_results
    result_logged: bool
