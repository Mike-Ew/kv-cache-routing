"""Six routing policies for the KV Cache Router.

Each policy implements: route(request, instance_states, cache_affinity) -> instance_index
"""

import hashlib
import random
from abc import ABC, abstractmethod


def _stable_hash(key: str) -> int:
    """Deterministic hash that is stable across Python process restarts."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16)


class Policy(ABC):
    """Base class for routing policies."""

    @abstractmethod
    def route(self, request: dict, instance_states: list[dict], cache_affinity: list[float]) -> int:
        ...


class RoundRobin(Policy):
    """Cycle through instances. Zero intelligence."""

    def __init__(self, n_instances: int):
        self.n = n_instances
        self.counter = 0

    def route(self, request, instance_states, cache_affinity):
        idx = self.counter % self.n
        self.counter += 1
        return idx


class JoinShortestQueue(Policy):
    """Pick the instance with the fewest waiting requests."""

    def route(self, request, instance_states, cache_affinity):
        min_q = min(s["queue_depth"] for s in instance_states)
        tied = [i for i, s in enumerate(instance_states) if s["queue_depth"] == min_q]
        return random.choice(tied)


class PowerOfTwoChoices(Policy):
    """Sample 2 random instances, pick the one with the shorter queue."""

    def route(self, request, instance_states, cache_affinity):
        n = len(instance_states)
        if n < 2:
            return 0
        a, b = random.sample(range(n), 2)
        return a if instance_states[a]["queue_depth"] <= instance_states[b]["queue_depth"] else b


class SessionAffinityHash(Policy):
    """Hash session/user ID to a fixed instance.

    All turns of a conversation go to the same place.
    Uses stable hashing for reproducibility across process restarts.
    """

    def route(self, request, instance_states, cache_affinity):
        session_id = request.get("user", request.get("session_id", "default"))
        return _stable_hash(session_id) % len(instance_states)


class PrefixKeyAffinity(Policy):
    """Hash the system prompt to an instance; fall back to JSQ if overloaded."""

    def __init__(self, n_instances: int, queue_threshold: int = 10):
        self.n = n_instances
        self.threshold = queue_threshold

    def route(self, request, instance_states, cache_affinity):
        messages = request.get("messages", [])
        if messages and messages[0].get("role") == "system":
            prefix_key = messages[0]["content"][:200]
        elif messages:
            prefix_key = messages[0]["content"][:200]
        else:
            prefix_key = "default"

        preferred = _stable_hash(prefix_key) % self.n
        if instance_states[preferred]["queue_depth"] < self.threshold:
            return preferred
        # Fallback to JSQ with tie-breaking
        min_q = min(s["queue_depth"] for s in instance_states)
        tied = [i for i in range(self.n) if instance_states[i]["queue_depth"] == min_q]
        return random.choice(tied)


class LoadCacheAwareScoring(Policy):
    """Score = alpha * cache_affinity - beta * load_pressure.

    The main contribution. Balances cache reuse against queue depth.
    Ties are broken randomly to avoid hot-spotting instance 0.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta

    def route(self, request, instance_states, cache_affinity):
        max_q = max(s["queue_depth"] for s in instance_states) or 1
        scores = []
        for i, state in enumerate(instance_states):
            cache_score = cache_affinity[i] if i < len(cache_affinity) else 0.0
            load_score = state["queue_depth"] / max_q
            score = self.alpha * cache_score - self.beta * load_score
            scores.append(score)
        max_score = max(scores)
        tied = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(tied)


# Registry for easy lookup by name
POLICIES = {
    "round_robin": RoundRobin,
    "jsq": JoinShortestQueue,
    "p2c": PowerOfTwoChoices,
    "session_affinity": SessionAffinityHash,
    "prefix_affinity": PrefixKeyAffinity,
    "load_cache_aware": LoadCacheAwareScoring,
}
