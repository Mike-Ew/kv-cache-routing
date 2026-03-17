# AI Collaboration Guide (Shared)

This repo currently contains the project specification for **Cache-Aware Routing for Multi-Instance LLM Serving**. The codebase described in `docs/project_plan_final.md` is not implemented yet.

Use this document as the shared contract for how Gemini, Claude, and Codex collaborate without stepping on each other.

## Goal (What We Are Building)

- Two (or more) `vllm serve` instances running the same model with prefix caching enabled.
- A **LangGraph-based Agentic Router** that receives OpenAI-compatible chat completion requests and routes each request to one instance using a pluggable routing policy node.
- A metrics pipeline that scrapes vLLM's `/metrics` Prometheus endpoint to measure **TTFT**, **cache hit rate**, **KV tokens computed**, and **queue depth**.
- Workload generators (ShareGPT replay + synthetic RAG) and analysis scripts to produce paper-ready plots.

## Working Style (Step-By-Step, No Rushing)

- One milestone at a time: align interface decisions first, then implement the smallest runnable slice, then iterate.
- Every change should be easy to verify: include a command to run and what "success" looks like.
- Prefer boring, reproducible engineering over cleverness.

## Source Of Truth

- Project intent and evaluation plan: `docs/project_plan_final.md`
- High-level overview: `README.md`
- If a decision impacts multiple components, record it in this file under "Decisions".

## Decisions (Record Here)

Fill these in as we decide them. Keep answers short and concrete.

- **Request format:** OpenAI `POST /v1/chat/completions` compatible (yes/no, and which fields we require).
- **Router forwarding:** `aiohttp` vs `httpx` (and timeouts/retries policy).
- **Config format:** YAML (recommended) vs JSON; where configs live (e.g., `configs/`).
- **Results format:** JSONL recommended (one row per request) + a summary JSON per run.
- **Experiment IDs:** naming scheme, required metadata (git SHA, seed, hardware, model, policy, load).
- **Metric names:** exact vLLM Prometheus metric keys we will treat as canonical.

## Handoff Format (Required For Cross-AI Work)

When handing work to another AI (or to David), use this template:

- **Goal:** what problem is being solved.
- **Change summary:** files/sections touched and why.
- **How to run:** exact commands.
- **Expected output:** key log lines, files produced, or plot names.
- **Assumptions/risks:** anything that could break on another machine.

If you cannot provide commands (e.g., writing-only task), provide:

- **Where this text goes:** exact file path and section header.

## Milestones (Definition Of Done)

### M0: Repo Bootstrap

- Minimal Python project structure exists (router/workloads/analysis folders).
- `README.md` and docs describe how to run the first smoke test.

### M1: Minimal Router (Round-Robin)

- **Graph Nodes:** `scrape_metrics`, `update_cache_dir`, `score_and_route`, `forward_request`, `record_results`.
- **StateGraph:** Nodes wired together in `graph.py`.
- A single `curl` request returns a model response through the LangGraph router.

### M2: Metrics + Instance State

- Router can scrape `/metrics` from each vLLM instance and maintain an in-memory view of:
  - `queue_depth` (waiting requests)
  - `prefix_cache_hit_rate` (or nearest available metric)
  - TTFT summary statistics (if exposed)

### M3: Policies

- Implement the 6 policies from the plan (RR, JSQ, P2C, Session Affinity, Prefix-Key Affinity, Load+Cache scoring).
- Policy selection is controlled by config.

### M4: Workloads + Results

- A small synthetic workload runner exists for smoke tests.
- ShareGPT replay and synthetic RAG workload can run at a controlled request rate.
- Each run produces JSONL results + a summary report.

### M5: Plots + Paper Artifacts

- Scripts generate: TTFT CDF, cache hit bars, alpha/beta heatmap, staleness curve, scaling plot.
- Figure filenames and captions are stable and referenced in the paper.

## Guardrails (Avoid Rework)

- Do not change request schemas or config keys without updating the shared "Decisions" section and calling out breaking changes.
- Do not over-claim performance: focus on **relative comparisons** and document the 1-GPU/2-instance contention caveat.
- Prefer "Option C" cache directory (track routing decisions) first; only add engine modifications if we explicitly agree.
