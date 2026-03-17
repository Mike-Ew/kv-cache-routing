# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research project evaluating six gateway-level routing policies for multi-instance vLLM serving, studying the tradeoff between KV cache reuse and load balance. This is a **measurement and empirical systems paper** — not a new algorithm invention. Focus on **relative comparisons** since both vLLM instances share one GPU.

**Current status:** Stage 0 (Repo Bootstrap) — planning docs exist, no implementation code yet. See `docs/implementation/stages.md` for milestone tracking.

## Hardware Setup

- 1x NVIDIA RTX 5080 (16 GB VRAM)
- 2x vLLM instances running `unsloth/Llama-3.2-3B-Instruct` at 0.45 GPU memory utilization each
- Both instances share one GPU — absolute numbers have contention; all analysis must focus on relative policy comparisons

### Launching vLLM Instances

```bash
# Instance 0
CUDA_VISIBLE_DEVICES=0 vllm serve unsloth/Llama-3.2-3B-Instruct \
  --port 8000 --gpu-memory-utilization 0.45 --enable-prefix-caching --disable-log-requests

# Instance 1
CUDA_VISIBLE_DEVICES=0 vllm serve unsloth/Llama-3.2-3B-Instruct \
  --port 8001 --gpu-memory-utilization 0.45 --enable-prefix-caching --disable-log-requests
```

## Dependencies

```bash
pip install vllm aiohttp fastapi uvicorn          # Core
pip install matplotlib numpy pandas seaborn        # Analysis
pip install datasets                                # HuggingFace datasets
```

## Architecture (Planned)

The system is a FastAPI gateway that sits in front of vLLM instances:

- **`router/`** — FastAPI gateway (`server.py`), pluggable policies (`policies.py`), cache directory (`cache_directory.py`), metrics collector (`metrics_collector.py`)
- **`workloads/`** — ShareGPT multi-turn replayer, synthetic RAG generator, mixed workload
- **`simulator/`** — Trace-driven scaling simulator with LRU cache model, calibrated from real measurements. Used for N=4,8,16 scaling experiments
- **`analysis/`** — Plotting scripts (TTFT CDF, cache heatmap, staleness curves, scaling plots)
- **`scripts/`** — Shell helpers to launch instances and run experiments
- **`configs/`** — Per-experiment YAML configurations

### Key Design Decisions

- **Request format:** OpenAI-compatible `POST /v1/chat/completions`
- **Policy interface:** `Policy.route(request, states) -> instance_index`
- **Cache directory:** Start with Option C (track routing decisions at the proxy, assume cache presence) — no vLLM engine modifications
- **Metrics source:** vLLM's native Prometheus `/metrics` endpoint (not proxy timing) for TTFT, cache hit rate, KV tokens computed, queue depth
- **Results format:** JSONL (one row per request) + summary JSON per run
- **Async I/O:** Use `aiohttp` or `httpx` with explicit timeouts for router forwarding and metrics scraping

### Six Routing Policies

| Policy                   | Type        | Key Logic                                              |
|--------------------------|-------------|--------------------------------------------------------|
| Round-Robin              | Load-only   | Cycle through instances                                |
| Join-Shortest-Queue      | Load-only   | Pick instance with fewest waiting requests             |
| Power-of-Two-Choices     | Load-only   | Sample 2, pick less busy                               |
| Session Affinity Hash    | Cache-aware | Hash session/user ID to instance                       |
| Prefix-Key Affinity      | Cache-aware | Hash system prompt to instance; fall back to JSQ       |
| Load+Cache-Aware Scoring | Cache-aware | `score = α * cache_affinity - β * load_pressure` (main contribution) |

## Multi-AI Collaboration

This repo uses three AI agents with distinct roles. See role-specific docs for scope boundaries:

- **Claude** (`docs/AI_CLAUDE.md`): Methodology, writing, claim discipline — ensures evaluation is credible and paper is honest
- **Codex** (`docs/AI_CODEX.md`): Implementation and integration — turns plan into runnable code
- **Gemini** (`docs/AI_GEMINI.md`): Workloads, analysis, visuals — creates workloads and produces plots

Shared contract and handoff format: `docs/AI_SHARED.md`. Cross-agent decisions must be recorded in the "Decisions" section there.

## Key Source of Truth Files

- `docs/project_plan_final.md` — Complete project plan with reference implementations, experiment design, paper outline
- `docs/AI_SHARED.md` — Shared decisions and milestone definitions
- `docs/implementation/stages.md` — Implementation progress tracking

## Claim Discipline

- Always restate the 2-instances-on-1-GPU caveat
- Avoid "state of the art" language — this is a measurement study
- Do not claim to outperform systems that modify the engine (Mooncake, Preble)
- If results are mixed, say so and describe boundary conditions
