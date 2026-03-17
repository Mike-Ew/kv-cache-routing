"""LangGraph node functions for the routing pipeline."""

import os
import re
import time
import json
import random
import logging
import aiohttp
from pathlib import Path
from router.state import RouterState
from router.cache_directory import CacheDirectory

logger = logging.getLogger("router")

# Mock mode: set MOCK=1 to test without vLLM running
MOCK_MODE = os.environ.get("MOCK", "0") == "1"

# vLLM instance URLs (WSL via Tailscale)
VLLM_INSTANCES = [
    "http://100.106.110.117:8000",
    "http://100.106.110.117:8001",
]

# Shared cache directory (persists across requests)
_staleness = float(os.environ.get("CACHE_STALENESS_SEC", "0"))
cache_dir = CacheDirectory(n_instances=len(VLLM_INSTANCES), staleness_sec=_staleness)
if _staleness > 0:
    logger.info(f"Cache directory staleness: {_staleness}s")

# Active routing policy — set via ROUTING_POLICY env var
from router.policies import POLICIES
_policy_name = os.environ.get("ROUTING_POLICY", "round_robin")
if _policy_name not in POLICIES:
    _valid = ", ".join(POLICIES.keys())
    raise ValueError(f"Unknown ROUTING_POLICY='{_policy_name}'. Valid options: {_valid}")
_PolicyClass = POLICIES[_policy_name]
# Policies that need n_instances in their constructor
if _policy_name in ("round_robin", "prefix_affinity"):
    active_policy = _PolicyClass(n_instances=len(VLLM_INSTANCES))
elif _policy_name == "load_cache_aware":
    _alpha = float(os.environ.get("LCA_ALPHA", "0.7"))
    _beta = float(os.environ.get("LCA_BETA", "0.3"))
    active_policy = _PolicyClass(alpha=_alpha, beta=_beta)
    logger.info(f"LoadCacheAware: alpha={_alpha}, beta={_beta}")
else:
    active_policy = _PolicyClass()
logger.info(f"Active routing policy: {_policy_name}")

RESULTS_DIR = Path(__file__).parent.parent / "results"


async def scrape_metrics(state: RouterState) -> dict:
    """Node 1: Poll /metrics from each vLLM instance."""
    if MOCK_MODE:
        return {"instance_states": [
            {"queue_depth": random.randint(0, 5), "num_running": 0,
             "cache_hit_rate": random.random(), "avg_ttft": 0.0},
            {"queue_depth": random.randint(0, 5), "num_running": 0,
             "cache_hit_rate": random.random(), "avg_ttft": 0.0},
        ]}

    instance_states = []
    async with aiohttp.ClientSession() as session:
        for url in VLLM_INSTANCES:
            try:
                async with session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    text = await resp.text()

                    queue_depth = _parse_metric(text, "vllm:num_requests_waiting")
                    num_running = _parse_metric(text, "vllm:num_requests_running")

                    # Cache hit rate = hits / queries (both are cumulative counters)
                    cache_hits = _parse_metric(text, "vllm:prefix_cache_hits_total")
                    cache_queries = _parse_metric(text, "vllm:prefix_cache_queries_total")
                    cache_hit_rate = cache_hits / cache_queries if cache_queries > 0 else 0.0

                    # Average TTFT = sum / count
                    ttft_sum = _parse_metric(text, "vllm:time_to_first_token_seconds_sum")
                    ttft_count = _parse_metric(text, "vllm:time_to_first_token_seconds_count")
                    avg_ttft = ttft_sum / ttft_count if ttft_count > 0 else 0.0

                    instance_states.append({
                        "queue_depth": int(queue_depth),
                        "num_running": int(num_running),
                        "cache_hit_rate": round(cache_hit_rate, 4),
                        "avg_ttft": round(avg_ttft, 4),
                    })
            except Exception as e:
                logger.warning(f"Failed to scrape metrics from {url}: {e}")
                # Mark unreachable instances with high queue depth to repel traffic
                instance_states.append({
                    "queue_depth": 999, "num_running": 0,
                    "cache_hit_rate": 0.0, "avg_ttft": 0.0,
                    "unreachable": True,
                })

    logger.info(f"Metrics: {instance_states}")
    return {"instance_states": instance_states}


async def update_cache_directory(state: RouterState) -> dict:
    """Node 2: Look up cache affinity for this request."""
    request = state["request"]
    prefix_key = CacheDirectory.extract_prefix_key(request)
    session_id = CacheDirectory.extract_session_id(request)
    affinity = cache_dir.get_affinity(prefix_key, session_id)
    logger.info(f"Cache affinity: {affinity} (prefix='{prefix_key[:40]}...', session='{session_id}')")
    return {"cache_affinity": affinity}


async def score_and_route(state: RouterState) -> dict:
    """Node 3: Run the active routing policy and pick an instance."""
    chosen = active_policy.route(
        state["request"],
        state["instance_states"],
        state["cache_affinity"],
    )
    logger.info(f"Policy '{_policy_name}' chose instance {chosen}")
    return {"chosen_instance": chosen}


async def forward_request(state: RouterState) -> dict:
    """Node 4: Send the request to the chosen vLLM instance."""
    instance_url = VLLM_INSTANCES[state["chosen_instance"]]
    request_body = state["request"]

    if MOCK_MODE:
        user_msg = ""
        for msg in request_body.get("messages", []):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
        data = {
            "id": f"mock-{int(time.time())}",
            "object": "chat.completion",
            "model": request_body.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"[MOCK] Echo: {user_msg}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_routing_metadata": {
                "instance": state["chosen_instance"],
                "instance_url": instance_url,
                "proxy_latency_ms": round(random.uniform(20, 80), 1),
            },
        }
        return {"response": data}

    try:
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            async with session.post(
                f"{instance_url}/v1/chat/completions",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                elapsed = time.monotonic() - start

                if resp.status != 200:
                    logger.error(f"Instance {instance_url} returned HTTP {resp.status}: {data}")

        data["_routing_metadata"] = {
            "instance": state["chosen_instance"],
            "instance_url": instance_url,
            "proxy_latency_ms": round(elapsed * 1000, 1),
        }
        return {"response": data}
    except Exception as e:
        logger.error(f"Failed to forward request to {instance_url}: {e}")
        return {"response": {
            "error": {"message": f"Backend {instance_url} unreachable: {e}", "type": "proxy_error"},
            "_routing_metadata": {
                "instance": state["chosen_instance"],
                "instance_url": instance_url,
                "proxy_latency_ms": -1,
            },
        }}


async def record_results(state: RouterState) -> dict:
    """Node 5: Log the request outcome and update the cache directory."""
    # Update cache directory with this routing decision
    request = state["request"]
    prefix_key = CacheDirectory.extract_prefix_key(request)
    session_id = CacheDirectory.extract_session_id(request)
    cache_dir.record(prefix_key, session_id, state["chosen_instance"])

    # Log to JSONL
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = state["response"].get("_routing_metadata", {})
    record = {
        "timestamp": time.time(),
        "instance": state["chosen_instance"],
        "proxy_latency_ms": metadata.get("proxy_latency_ms"),
        "model": state["request"].get("model"),
        "cache_affinity": state.get("cache_affinity", []),
        "instance_states": state.get("instance_states", []),
    }
    results_file = RESULTS_DIR / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"result_logged": True}


def _parse_metric(text: str, name: str) -> float:
    """Extract a Prometheus metric value from vLLM /metrics text."""
    pattern = rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([\d.eE+\-]+)'
    for line in text.split("\n"):
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return 0.0
