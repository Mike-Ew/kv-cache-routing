"""Cache directory: tracks which prefixes were routed to which instance.

This is "Option C" from the project plan — we infer cache state from
routing history rather than querying vLLM internals. When we route a
request with prefix P to instance I, we record that. Next time a request
with the same prefix arrives, we know instance I probably has it cached.
"""

import time


class CacheDirectory:
    """Maps prefix keys to the instance that most recently received them."""

    def __init__(self, n_instances: int, staleness_sec: float = 0.0):
        self.n = n_instances
        self.staleness_sec = staleness_sec  # 0 = no staleness (always fresh)
        self.prefix_map: dict[str, tuple[int, float]] = {}   # key -> (instance, timestamp)
        self.session_map: dict[str, tuple[int, float]] = {}   # session -> (instance, timestamp)

    def get_affinity(self, prefix_key: str, session_id: str) -> list[float]:
        """Return a cache affinity score for each instance.

        1.0 if the instance has the prefix cached (by routing history),
        0.5 if it has the session (weaker signal),
        0.0 otherwise.

        Entries older than staleness_sec are treated as stale (ignored).
        """
        now = time.monotonic()
        scores = [0.0] * self.n

        if prefix_key in self.prefix_map:
            inst, ts = self.prefix_map[prefix_key]
            if self.staleness_sec <= 0 or (now - ts) <= self.staleness_sec:
                scores[inst] = 1.0

        if session_id in self.session_map:
            inst, ts = self.session_map[session_id]
            if self.staleness_sec <= 0 or (now - ts) <= self.staleness_sec:
                scores[inst] = max(scores[inst], 0.5)

        return scores

    def record(self, prefix_key: str, session_id: str, instance_id: int):
        """Record that a request was routed to an instance."""
        now = time.monotonic()
        self.prefix_map[prefix_key] = (instance_id, now)
        if session_id:
            self.session_map[session_id] = (instance_id, now)

    @staticmethod
    def extract_prefix_key(request: dict) -> str:
        """Extract a cache key from the request's message prefix."""
        messages = request.get("messages", [])
        if messages and messages[0].get("role") == "system":
            return messages[0]["content"][:200]
        elif messages:
            return messages[0]["content"][:200]
        return "default"

    @staticmethod
    def extract_session_id(request: dict) -> str:
        """Extract a session/user ID from the request."""
        return request.get("user", request.get("session_id", ""))
