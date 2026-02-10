# Final Project Plan: Cache-Aware Gateway Routing for Multi-Instance LLM Serving

## A Reproducible Evaluation of Locality vs. Load Balance Tradeoffs

---

## 1. One-Paragraph Pitch

Large language model serving deployments typically run multiple inference engine instances behind a load balancer. Each instance maintains its own KV cache of previously computed attention states; routing a request to an instance that already holds relevant cached prefixes can eliminate redundant computation and dramatically reduce time-to-first-token (TTFT). Recent systems such as Preble, DualMap, and Ray Serve's `PrefixCacheAffinityRouter` have demonstrated this value, but embed routing logic tightly within custom schedulers, making it difficult to isolate the routing decision from other optimizations. We implement and evaluate six gateway-level routing strategies — round-robin, join-shortest-queue, power-of-two-choices, session-affinity hashing, prefix-key affinity, and a novel load+cache-aware scoring policy — across multiple vLLM instances replaying real conversation traces. We measure ground-truth metrics via vLLM's native Prometheus endpoint, provide the first systematic study of cache directory staleness, and release an open-source implementation for reproducibility.

---

## 2. Research Questions

**RQ1:** Under what workload characteristics (multi-turn depth, shared-prefix rate, prompt length) does cache-aware routing improve TTFT and tail latency versus load-only routing?

**RQ2:** What is the tradeoff surface between cache affinity and queueing delay, and can a simple α/β scoring policy find robust settings across workloads?

**RQ3:** How sensitive are benefits to directory staleness (cache index refresh interval), and what refresh overhead is tolerable?

---

## 3. Novelty Framing

**Do NOT say:** "No comparison of cache-aware routing exists."

**DO say:**

> Recent systems such as Preble, DualMap, and Ray Serve's PrefixCacheAffinityRouter have demonstrated the value of cache-aware scheduling. However, these systems embed routing logic tightly within custom schedulers or specific orchestration frameworks, making it difficult to isolate the routing decision from other optimizations.
>
> We provide three contributions:
>
> 1. A **reproducible, apples-to-apples evaluation** of six gateway-level routing policies for vanilla multi-instance vLLM, using vLLM's native metrics for ground-truth TTFT and cache reuse measurements.
> 2. A **systematic study of cache directory staleness** — the first to parameter-sweep refresh intervals and quantify the staleness-vs-overhead tradeoff for prefix-aware routing.
> 3. A **simple, tunable scoring policy** (α/β) that requires no engine modification and is robust across workload types.

---

## 4. Hardware Setup: Making 1× RTX 5080 Work

![System Architecture](image.png)

### Launch commands

```bash
# Instance 0
CUDA_VISIBLE_DEVICES=0 vllm serve unsloth/Llama-3.2-3B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.45 \
  --enable-prefix-caching \
  --disable-log-requests

# Instance 1
CUDA_VISIBLE_DEVICES=0 vllm serve unsloth/Llama-3.2-3B-Instruct \
  --port 8001 \
  --gpu-memory-utilization 0.45 \
  --enable-prefix-caching \
  --disable-log-requests
```

### Important caveat for the paper

Both instances share one GPU, so absolute throughput numbers have GPU contention. Address this honestly:

> "Our two-instance prototype shares a single GPU, introducing contention absent in production multi-GPU deployments. We therefore focus on **relative** comparisons between policies (all experience the same contention) and validate scaling trends with a trace-driven simulator calibrated to per-instance latency profiles."

---

## 5. The 6 Routing Policies

### Load-Only Baselines (isolate load-balancing gains)

```python
class RoundRobin:
    """Cycle through instances. Zero intelligence."""
    def __init__(self, n):
        self.i = 0
        self.n = n
    def route(self, request, states):
        idx = self.i % self.n
        self.i += 1
        return idx

class JoinShortestQueue:
    """Always pick the instance with the fewest waiting requests."""
    def route(self, request, states):
        return min(range(len(states)), key=lambda i: states[i].queue_depth)

class PowerOfTwoChoices:
    """Sample 2 random instances, pick the one with shorter queue."""
    def route(self, request, states):
        a, b = random.sample(range(len(states)), 2)
        return a if states[a].queue_depth <= states[b].queue_depth else b
```

### Cache-Aware Policies (your focus)

```python
class SessionAffinityHash:
    """
    Hash session/user ID to a fixed instance.
    All turns of a conversation go to the same place.
    NOT consistent hashing — just modular hash routing.
    """
    def route(self, request, states):
        session_id = request.get("user", request.get("session_id", "default"))
        return hash(session_id) % len(states)

class PrefixKeyAffinity:
    """
    Hash the first K tokens (or system prompt template) to an instance.
    Captures cross-user shared prefix locality.
    Override to JSQ if target instance is overloaded (queue > threshold).
    """
    def __init__(self, n_instances, queue_threshold=10):
        self.n = n_instances
        self.threshold = queue_threshold

    def route(self, request, states):
        prefix_key = self._extract_prefix_key(request)
        preferred = hash(prefix_key) % self.n
        if states[preferred].queue_depth < self.threshold:
            return preferred
        return min(range(self.n), key=lambda i: states[i].queue_depth)

    def _extract_prefix_key(self, request):
        messages = request.get("messages", [])
        if messages and messages[0]["role"] == "system":
            return messages[0]["content"][:200]
        elif messages:
            return messages[0]["content"][:200]
        return "default"

class LoadCacheAwareScoring:
    """
    YOUR MAIN CONTRIBUTION.
    Score = α * cache_affinity - β * load_pressure
    Sweep α and β to find the Pareto frontier.
    """
    def __init__(self, n_instances, alpha=0.7, beta=0.3):
        self.n = n_instances
        self.alpha = alpha
        self.beta = beta
        self.directory = {}        # prefix_key -> instance_id
        self.session_map = {}      # session_id -> instance_id

    def route(self, request, states):
        prefix_key = self._extract_prefix_key(request)
        session_id = request.get("user", "default")

        scores = []
        for i, state in enumerate(states):
            cache_score = 0.0
            if self.directory.get(prefix_key) == i:
                cache_score = 1.0
            elif self.session_map.get(session_id) == i:
                cache_score = 0.8
            max_q = max(s.queue_depth for s in states) or 1
            load_score = state.queue_depth / max_q
            score = self.alpha * cache_score - self.beta * load_score
            scores.append(score)

        chosen = scores.index(max(scores))
        self.directory[prefix_key] = chosen
        self.session_map[session_id] = chosen
        return chosen

    def _extract_prefix_key(self, request):
        messages = request.get("messages", [])
        if messages and messages[0]["role"] == "system":
            return messages[0]["content"][:200]
        elif messages:
            return messages[0]["content"][:200]
        return "default"
```

---

## 6. Metrics Collection: Use vLLM's /metrics Endpoint

**This is critical.** Don't rely only on proxy timing.

```python
import aiohttp
import re

async def scrape_vllm_metrics(instance_url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{instance_url}/metrics") as resp:
            text = await resp.text()

    metrics = {}
    ttft_sum = _parse_metric(text, "vllm:time_to_first_token_seconds_sum")
    ttft_count = _parse_metric(text, "vllm:time_to_first_token_seconds_count")
    if ttft_count and ttft_count > 0:
        metrics["avg_ttft"] = ttft_sum / ttft_count
    metrics["prefix_cache_hit_rate"] = _parse_metric(text, "vllm:prefix_cache_hit_rate")
    metrics["prefill_kv_computed"] = _parse_metric(text, "vllm:request_prefill_kv_computed_tokens_sum")
    metrics["waiting"] = _parse_metric(text, "vllm:num_requests_waiting")
    metrics["running"] = _parse_metric(text, "vllm:num_requests_running")
    return metrics

def _parse_metric(text: str, name: str) -> float:
    pattern = rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([\d.eE+\-]+)'
    for line in text.split('\n'):
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return 0.0
```

### Metrics to report in the paper

| Metric               | Source                                    | Why it matters                          |
| -------------------- | ----------------------------------------- | --------------------------------------- |
| TTFT (P50, P95, P99) | `vllm:time_to_first_token_seconds`        | Primary metric; shows cache benefit     |
| Cache hit rate       | `vllm:prefix_cache_hit_rate`              | Direct measure of routing effectiveness |
| KV tokens computed   | `vllm:request_prefill_kv_computed_tokens` | Shows how much compute was saved        |
| Queue depth          | `vllm:num_requests_waiting`               | Load balance quality                    |
| End-to-end latency   | Proxy-side (for completeness)             | User-facing metric                      |
| Throughput (req/s)   | Proxy-side counter                        | System capacity                         |
| Load imbalance ratio | max queue / avg queue                     | Fairness across instances               |

---

## 7. Cache Index Implementation Strategy

Three options in order of complexity:

| Option             | Approach                                                                                                                 | Accuracy | Effort | Recommendation           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ | -------- | ------ | ------------------------ |
| **C (Start here)** | Track routing decisions at the proxy. If you sent prefix P to instance 2, assume instance 2 has P cached until eviction. | ~80%     | Low    | **Use this first**       |
| **A**              | After each request, instance reports cached prefix hashes to the router.                                                 | ~95%     | Medium | Upgrade if time permits  |
| **B**              | Poll vLLM's internal cache state via a custom endpoint.                                                                  | ~100%    | High   | Out of scope for 4 weeks |

Start with Option C — it is surprisingly effective for evaluation purposes and avoids the complexity of modifying vLLM internals.

---

## 8. Workload Preparation

### ShareGPT Multi-Turn Traces

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

```python
# workloads/sharegpt_replayer.py
import json, asyncio, aiohttp, time, random

async def replay_conversation(session, router_url, conversation, conv_id, rate):
    """Replay one multi-turn conversation."""
    messages = []
    results = []
    for turn in conversation:
        messages.append({"role": turn["from"], "content": turn["value"]})
        start = time.monotonic()
        async with session.post(f"{router_url}/v1/chat/completions", json={
            "model": "unsloth/Llama-3.2-3B-Instruct",
            "messages": messages,
            "max_tokens": 256,
            "user": conv_id  # for session affinity routing
        }) as resp:
            data = await resp.json()
            ttft = data.get("ttft", time.monotonic() - start)
            results.append({"ttft": ttft, "turn": len(messages)})
        await asyncio.sleep(random.expovariate(rate))
    return results

async def run_workload(router_url, dataset_path, n_conversations, rate):
    with open(dataset_path) as f:
        conversations = json.load(f)
    sampled = random.sample(conversations, n_conversations)
    async with aiohttp.ClientSession() as session:
        tasks = [replay_conversation(session, router_url, c, str(i), rate)
                 for i, c in enumerate(sampled)]
        return await asyncio.gather(*tasks)
```

### Synthetic RAG Workload

Fixed system prompt (500 tokens) + varying user queries from different "users" — tests cross-user shared prefix locality.

---

## 9. Trace-Driven Simulator (for Scaling to N=4, 8, 16)

Since you have 1 GPU with 2 real instances, use a simulator for scaling experiments. Calibrate with measurements from the real setup.

```python
from collections import OrderedDict
from types import SimpleNamespace

class CacheSimulator:
    def __init__(self, n_instances, cache_capacity_per_instance,
                 ttft_cache_hit_ms, ttft_cache_miss_ms_per_token):
        self.n = n_instances
        self.caches = [LRUCache(cache_capacity_per_instance) for _ in range(n_instances)]
        self.queues = [0] * n_instances
        self.ttft_hit = ttft_cache_hit_ms
        self.ttft_miss_per_token = ttft_cache_miss_ms_per_token

    def process_request(self, request, policy):
        prefix_key = request["prefix_key"]
        prefix_len = request["prefix_length"]
        instance_id = policy.route(request, self._get_states())
        cached_len = self.caches[instance_id].lookup(prefix_key)
        tokens_to_compute = prefix_len - cached_len
        if tokens_to_compute == 0:
            ttft = self.ttft_hit
        else:
            ttft = self.ttft_hit + tokens_to_compute * self.ttft_miss_per_token
        ttft += self.queues[instance_id] * self.ttft_hit
        self.caches[instance_id].insert(prefix_key, prefix_len)
        self.queues[instance_id] += 1
        return {"instance": instance_id, "ttft_ms": ttft,
                "cache_hit_tokens": cached_len, "computed_tokens": tokens_to_compute}

    def _get_states(self):
        return [SimpleNamespace(queue_depth=q) for q in self.queues]

class LRUCache:
    def __init__(self, capacity_tokens):
        self.capacity = capacity_tokens
        self.entries = OrderedDict()
        self.used = 0

    def lookup(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        return 0

    def insert(self, key, length):
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        while self.used + length > self.capacity and self.entries:
            _, evicted_len = self.entries.popitem(last=False)
            self.used -= evicted_len
        self.entries[key] = length
        self.used += length
```

### How to calibrate the simulator

```python
# Measure baseline TTFT for various prefix lengths with cache hit vs miss
test_prefix_lengths = [128, 256, 512, 1024, 2048]

for length in test_prefix_lengths:
    ttft_miss = send_request_and_measure(prefix_tokens=length, cold=True)
    ttft_hit  = send_request_and_measure(prefix_tokens=length, cold=False)
    print(f"Prefix {length}: miss={ttft_miss:.1f}ms, hit={ttft_hit:.1f}ms, "
          f"speedup={ttft_miss/ttft_hit:.1f}x")
```

Report this calibration data in the paper — it's a contribution in itself.

---

## 10. Experiment Plan

### Experiment 1: Policy Comparison on Multi-Turn Chat

- **Workload:** ShareGPT multi-turn traces (filter to 2–10 turns, 100–2000 tokens/turn)
- **Setup:** 2 real vLLM instances
- **Policies:** All 6
- **Request rates:** 0.5×, 1×, 2× sustainable throughput
- **Key figure:** TTFT CDF for all 6 policies at 1× load
- **Expected finding:** Session affinity and Load+Cache-Aware beat load-only policies; JSQ beats round-robin but not cache-aware ones

### Experiment 2: RAG Workload (Shared System Prompts)

- **Workload:** Synthetic — fixed system prompt (500 tokens) + varying user queries from different "users"
- **Key figure:** Cache hit rate bar chart by policy
- **Expected finding:** PrefixKeyAffinity shines (cross-user prefix sharing); SessionAffinity does poorly (different users, same prefix)

### Experiment 3: α/β Sensitivity Sweep

- **Setup:** Load+Cache-Aware with α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, β ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- **Key figure:** Heatmap — color = P95 TTFT, axes = α, β
- **Secondary figure:** Pareto curve — cache hit rate vs. load imbalance
- **Expected finding:** Sweet spot around α=0.6–0.8, β=0.2–0.4

### Experiment 4: Directory Staleness (RQ3)

- **Setup:** Load+Cache-Aware, vary directory refresh interval (0ms/real-time, 100ms, 500ms, 1s, 5s, 30s, never)
- **Key figure:** Line plot — X = refresh interval, Y = cache hit rate
- **Expected finding:** Benefits plateau; refreshing every 1–5s is sufficient

### Experiment 5: Scaling (Simulator)

- **Setup:** Simulator calibrated from Exp 1–2 measurements
- **Scale:** N = 2, 4, 8, 16 instances
- **Key figure:** TTFT P95 vs. N for each policy
- **Expected finding:** Cache-aware routing becomes MORE important at larger N

---

## 11. Analysis & Essential Figures

```python
# analysis/plot_ttft_cdf.py
import matplotlib.pyplot as plt
import numpy as np

def plot_ttft_cdf(results_by_policy, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy_name, ttfts in results_by_policy.items():
        sorted_ttfts = np.sort(ttfts)
        cdf = np.arange(1, len(sorted_ttfts) + 1) / len(sorted_ttfts)
        ax.plot(sorted_ttfts * 1000, cdf, label=policy_name)
    ax.set_xlabel("Time to First Token (ms)")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
```

### Full figure list

1. **TTFT CDF plot** (one per workload): X=TTFT, Y=percentile, 6 lines (one per policy)
2. **Cache hit rate bar chart**: grouped by policy × workload
3. **Throughput vs. latency scatter**: each policy is a point
4. **α/β heatmap**: color = TTFT P95, axes = α and β values
5. **Staleness sensitivity line plot**: X=refresh interval, Y=cache hit rate
6. **Load imbalance box plot**: per policy
7. **Scaling line plot**: TTFT P95 vs. N instances (from simulator)

---

## 12. Paper Outline (8 pages, systems conference format)

### §1 Introduction (0.75 pages)

- LLM serving runs multiple instances; KV cache is per-instance
- Routing decision determines cache reuse (or waste)
- Recent work (Preble, DualMap) embeds routing in custom schedulers; we study the simpler question: what policy should a gateway use in front of unmodified vLLM instances?
- Contributions: (1) systematic eval, (2) staleness study, (3) simple tunable policy + open-source implementation

### §2 Background & Motivation (1 page)

- KV cache in transformers (brief)
- Prefix caching in vLLM (APC) and SGLang (RadixAttention)
- Why multi-instance is necessary (memory limits, throughput scaling, fault isolation)
- The routing problem: gateway sees requests, must pick instance
- Motivating measurement: cache hit vs miss TTFT on your hardware (from calibration data)
- Back-of-envelope: cache miss on a 2K token prefix costs ~50–200ms of redundant prefill

### §3 Routing Policy Design (1.5 pages)

- Describe all 6 policies with the load-only vs. cache-aware taxonomy
- Discuss the design space: load-only vs. cache-only vs. combined
- Explain the cache directory mechanism and its update protocol
- Discuss staleness: how often must the index refresh?

### §4 Experimental Setup (1 page)

- Hardware (1× RTX 5080, 2 vLLM instances)
- Model (Llama-3.2-3B-Instruct)
- Workloads (ShareGPT, synthetic RAG, mixed)
- Metrics (from vLLM `/metrics` — list exact metric names)
- Simulator design and calibration methodology
- Experiment methodology: cold start, warm-up, measure, repeat 3×

### §5 Evaluation (2.5 pages)

- Experiments 1–5 with 6–7 figures
- Each experiment tied to a specific RQ

### §6 Related Work (0.75 pages)

- Single-node KV management: vLLM, SGLang, Mooncake
- Distributed scheduling: Preble, DualMap, llm-d, SkyLB
- Classic load balancing: consistent hashing, JSQ, power-of-two-choices
- How we differ: gateway-level, no engine modification, staleness study

### §7 Discussion & Conclusion (0.5 pages)

- When cache-aware routing matters (long prefixes, multi-turn, high prefix overlap) vs. when it doesn't (single-turn, short prompts, very low load)
- Practical recommendation: use Load+Cache-Aware with α=X, β=Y as default
- Limitations (1 GPU, 3B model, simulated scaling — stated honestly)
- Future work: semantic similarity routing, cross-instance cache migration, adaptive policy switching

---

## 13. Week-by-Week Timeline

### Week 1: Infrastructure + Calibration

| Day | Task                                                                  |
| --- | --------------------------------------------------------------------- |
| 1–2 | Install vLLM, download Llama-3.2-3B-Instruct, verify 2-instance setup |
| 3–4 | Build router skeleton (FastAPI proxy with pluggable policy interface) |
| 5–6 | Build metrics collector (scrape `/metrics` from both instances)       |
| 7   | Run calibration benchmark (TTFT vs prefix length, hit vs miss)        |

### Week 2: Implement + Experiment

| Day | Task                                                                 |
| --- | -------------------------------------------------------------------- |
| 1–2 | Implement all 6 routing policies                                     |
| 3   | Prepare workloads (download ShareGPT, write synthetic RAG generator) |
| 4–5 | Run Experiments 1–3 (real instances)                                 |
| 6   | Run Experiment 4 (staleness sweep)                                   |
| 7   | Build and calibrate simulator, run Experiment 5                      |

### Week 3: Analysis + Paper Draft

| Day | Task                                                               |
| --- | ------------------------------------------------------------------ |
| 1–2 | Generate all figures (matplotlib)                                  |
| 3–5 | Write paper (Results first → System Design → Intro → Related Work) |
| 6–7 | Internal review — check flow, fill gaps, add missing analysis      |

### Week 4: Polish + Submission

| Day | Task                                                                               |
| --- | ---------------------------------------------------------------------------------- |
| 1–2 | Re-run any experiments where results were noisy or inconclusive                    |
| 3–4 | Final paper polish: proofread, check all figures have axis labels/legends/captions |
| 5   | Ensure reproducibility: document exact commands, seeds, hardware in README         |
| 6   | Clean up repo, write README, push to GitHub                                        |
| 7   | Final submission / buffer day                                                      |

---

## 14. Potential Pitfalls & How to Avoid Them

| Pitfall                                       | Solution                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| vLLM doesn't expose cache state easily        | Start with tracking-based approximation (Option C); upgrade to metrics-based (Option A) if time permits |
| TTFT is hard to measure through a proxy       | Use vLLM's native `/metrics` endpoint (engine-side TTFT) — way more credible than proxy timing          |
| ShareGPT has very long conversations          | Filter to conversations with 2–10 turns, 100–2000 tokens per turn                                       |
| Results are noisy                             | Run each experiment 3 times, report mean ± std; use fixed random seeds                                  |
| Load balancing effects only show at high load | Run at multiple request rates (0.5×, 1×, 2×, 4× sustainable throughput)                                 |
| Only 1 GPU, shared contention                 | Focus on **relative** comparisons (all policies share the same contention); use simulator for scaling   |
| Prefix cache eviction confounds results       | Set GPU memory utilization high within bounds (0.45 each) and use workloads where cache fits            |
| Python router becomes a bottleneck            | Use `asyncio`/`aiohttp` for async I/O; profile the router to ensure <1ms overhead vs. engine TTFT       |

---

## 15. How to Frame Your Contribution

**Do NOT claim:**

- "We invented a new routing algorithm" (consistent hashing and LPM are well-known)
- "Our system outperforms Mooncake/Preble/vLLM" (apples to oranges; they modify the engine)

**DO claim:**

- "We provide the first systematic, apples-to-apples comparison of cache-aware routing policies for multi-instance LLM serving under realistic workloads"
- "We identify the tradeoff between cache affinity and load balance, and show that a simple scoring policy navigates it effectively"
- "We quantify when cache-aware routing matters (long prefixes, high reuse) and when it doesn't (single-turn, low load)"
- "We are the first to study cache directory staleness and its impact on routing quality"
- "We release an open-source router framework for reproducibility"

This is a **measurement and empirical systems paper** — these are valued and publishable at workshops (MLSys workshop, SysML, HotNets) and second-tier conferences.

---

## 16. Stretch Goals (if you have extra time)

1. **Cross-instance cache migration:** When the best instance is overloaded, transfer KV cache to a less-loaded one via LMCache. Measure the break-even point (transfer cost vs. recompute cost).

2. **Adaptive policy switching:** Monitor workload characteristics in real-time and auto-switch between policies (e.g., use session affinity when load is low, switch to Load+Cache-Aware when load is high).

3. **Prefix deduplication analysis:** Analyze ShareGPT/LMSYS traces to quantify how much prefix overlap exists across different users. This motivates the whole paper and could be a standalone figure.

4. **Second model:** Repeat key experiments with Mistral-7B-v0.3 to show findings generalize across model architectures.

---

## 17. Software Requirements

```bash
# Core
pip install vllm aiohttp fastapi uvicorn

# Analysis
pip install matplotlib numpy pandas seaborn

# Dataset
pip install datasets  # for HuggingFace datasets
```

**Minimum viable setup:** 1 GPU with ≥16GB VRAM (RTX 5080), 32GB+ system RAM, 50GB free disk.

---

## 18. Repo Layout

```
kv-cache-routing/
├── README.md
├── router/
│   ├── server.py              # FastAPI gateway
│   ├── policies.py            # All 6 routing policies
│   ├── cache_directory.py     # Prefix key → instance mapping
│   └── metrics_collector.py   # Scrapes vLLM /metrics endpoints
├── workloads/
│   ├── sharegpt_replayer.py   # Replays multi-turn conversations
│   ├── rag_generator.py       # Synthetic RAG workload
│   └── mixed_generator.py     # 50/50 mix
├── simulator/
│   ├── cache_sim.py           # LRU cache simulator
│   ├── instance_sim.py        # Simulated vLLM instance
│   └── calibrate.py           # Measures real TTFT for calibration
├── analysis/
│   ├── plot_ttft_cdf.py
│   ├── plot_cache_heatmap.py
│   ├── plot_staleness.py
│   ├── plot_scaling.py
│   └── parse_metrics.py       # Parses collected Prometheus data
├── scripts/
│   ├── launch_instances.sh    # Starts 2 vLLM instances
│   ├── run_experiment.sh      # Runs one experiment end-to-end
│   └── run_all.sh             # Runs all experiments sequentially
├── configs/
│   ├── experiment1.yaml       # Per-experiment configurations
│   ├── experiment2.yaml
│   └── ...
└── paper/
    ├── main.tex
    └── figures/
```

---

## 19. Key References to Cite

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
2. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024
3. Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving," FAST 2025
4. Gao et al., "CachedAttention: Cost-efficient LLM Serving for Multi-turn Conversations," ATC 2024
5. Zhu et al., "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving," 2025
6. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024
7. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI 2024
8. Karger et al., "Consistent Hashing and Random Trees," STOC 1997
9. Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing," IEEE TPDS 2001
10. Preble — "Efficient Distributed Prompt Scheduling for LLM Serving" (for Session/Prefix-aware routing prior art)
11. Ray Serve `PrefixCacheAffinityRouter` (industry prior art)
