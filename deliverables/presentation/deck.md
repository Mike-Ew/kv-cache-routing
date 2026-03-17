---
marp: true
paginate: true
---

# Cache-Aware Gateway Routing
## For Multi-Instance LLM Serving

David Mike  
Distributed Systems  
Presentation due: Feb 26, 2026

---

## Problem

- Multi-instance LLM serving uses a load balancer / gateway.
- Each instance maintains its own KV cache (prefix caching).
- Routing impacts:
  - cache reuse (faster TTFT)
  - load balance (queueing delay)

---

## Key Idea

Routing should be cache-aware, not just load-aware.

- Best-case: send requests to instances with relevant cached prefixes.
- Risk: too much affinity can overload one instance.
- Goal: measure the tradeoff and find robust policies.

---

## System Architecture

- Workload generator sends chat requests.
- Router/gateway selects an instance using a policy.
- vLLM instances serve requests with prefix caching enabled.
- Metrics scraped from each instance `/metrics`.

Figure: `paper/figures/system_architecture.png`

---

## Routing Policies (6)

Load-only baselines:
- Round-robin
- Join-shortest-queue (JSQ)
- Power-of-two-choices (P2C)

Cache-aware:
- Session affinity hash
- Prefix-key affinity (system prompt / first-K tokens)
- Load + cache-aware scoring (alpha/beta)

---

## Research Questions

- RQ1: When does cache-aware routing help TTFT and tail latency?
- RQ2: What is the cache-vs-queue tradeoff surface (alpha/beta)?
- RQ3: How sensitive is routing to cache directory staleness?

---

## Hardware / Setup

- 1x RTX 5080 (16GB)
- 2x vLLM instances (shared GPU), prefix caching enabled
- Router: FastAPI with pluggable policy engine

Important caveat:
- 2 instances share 1 GPU; report relative comparisons; use simulator for scaling.

---

## Metrics (Ground Truth)

Scrape vLLM Prometheus `/metrics`:

- TTFT (P50/P95/P99): `vllm:time_to_first_token_seconds`
- Cache hit rate: `vllm:prefix_cache_hit_rate`
- KV tokens computed: `vllm:request_prefill_kv_computed_tokens`
- Queue depth: `vllm:num_requests_waiting`

---

## Workloads

- ShareGPT multi-turn replay:
  - captures within-session prefix reuse
- Synthetic RAG:
  - fixed system prompt + varied user queries
  - captures cross-user shared-prefix reuse

---

## Experiment Plan (High Level)

1. Policy comparison (TTFT CDF under multiple loads)
2. RAG workload (cache hit rate by policy)
3. Alpha/beta sweep (heatmap of P95 TTFT)
4. Directory staleness sweep (refresh interval vs hit rate)
5. Scaling via simulator (N = 2, 4, 8, 16)

---

## Expected Results / Hypotheses

- Cache-aware policies improve TTFT when prefixes are long and reused.
- Pure affinity can increase tail latency under load (queueing).
- Scoring policy finds a robust middle ground.
- Directory refresh every ~1–5s may be “good enough”.

---

## Status (Fill In Weekly)

Week 1:
- [ ] Router skeleton + RR working
- [ ] Metrics scraping working
- [ ] Smoke workload + JSONL results

Week 2:
- [ ] All 6 policies implemented
- [ ] ShareGPT + synthetic RAG workloads runnable
- [ ] First plots generated

---

## Risks + Mitigations

- Noisy results:
  - repeat runs, fixed seeds, warmups
- Router bottleneck:
  - async I/O, measure router overhead separately
- Cache-state visibility:
  - start with "Option C" (track routing decisions) first

---

## Appendix: What We Will Show

Figures:
- TTFT CDF (6 policies)
- Cache hit rate bars
- Alpha/beta heatmap
- Staleness curve
- Scaling curve (simulator)

---

# Q&A

