# Implementation Stages

Tracked milestones for the kv-cache-routing project, derived from the project plan.

---

## Stage 0 — Repo Bootstrap

- [ ] Project structure finalized
- [ ] README and docs describe how to run a smoke test
- [ ] Dependencies documented (`requirements.txt` or similar)

## Stage 1 — Minimal Router (Round-Robin)

- [ ] FastAPI endpoint accepts `POST /v1/chat/completions`
- [ ] Request forwarded to a vLLM instance via async HTTP
- [ ] Round-robin policy selects the instance
- [ ] Single `curl` request returns a model response through the router

## Stage 2 — Metrics + Instance State

- [ ] Router scrapes `/metrics` from each vLLM instance on an interval
- [ ] In-memory `InstanceState` tracks: queue depth, prefix cache hit rate, TTFT stats
- [ ] Metrics collection verified against live vLLM output

## Stage 3 — Routing Policies

- [ ] Policy interface defined: `Policy.route(request, states) → instance_index`
- [ ] Implement Join-Shortest-Queue (JSQ)
- [ ] Implement Power-of-Two-Choices (P2C)
- [ ] Implement Session Affinity Hash
- [ ] Implement Prefix-Key Affinity
- [ ] Implement Load+Cache-Aware Scoring (α/β)
- [ ] Policy selection controlled by config

## Stage 4 — Workloads + Results

- [ ] Synthetic smoke workload (no downloads, validates pipeline end-to-end)
- [ ] ShareGPT multi-turn replay (filtered: 2–10 turns, 100–2000 tokens/turn)
- [ ] Synthetic RAG workload (fixed system prompt + varying user queries)
- [ ] Each run produces `results.jsonl` + `summary.json`
- [ ] Experiment runner is config-driven with reproducible seeds

## Stage 5 — Analysis + Paper Artifacts

- [ ] TTFT CDF plot (per workload, 6 policy lines)
- [ ] Cache hit rate bar chart (policy × workload)
- [ ] α/β heatmap (P95 TTFT)
- [ ] Staleness sensitivity line plot
- [ ] Scaling line plot (simulator, N = 2, 4, 8, 16)
- [ ] All figures referenced in the paper with stable filenames

---

## Current Status

**Active Stage:** 0 — Repo Bootstrap
**Last Updated:** 2026-02-12
