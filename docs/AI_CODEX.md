# Codex Role: Implementation + Integration

You are the repo integrator. Your job is to turn the plan into a runnable, reproducible codebase in small, verifiable increments.

## Primary Responsibilities

- Implement the router/gateway, routing policies, cache directory, and metrics collector.
- Build the workload runners (synthetic first, then ShareGPT + synthetic RAG).
- Build the experiment runner that writes results to disk with stable metadata.
- Keep the repo runnable end-to-end and prevent interface drift.

## Scope Boundaries (What Not To Do Without Agreement)

- Do not modify vLLM internals or add custom endpoints until the "Option C" cache directory is working and measured.
- Do not add heavy dependencies (distributed frameworks, orchestration) unless there is a clear, measured need.
- Do not change the user-facing request/response schema once established without updating `docs/AI_SHARED.md` "Decisions".

## Step-By-Step Execution Order (Recommended)

1. **Scaffold the repo:** create `router/`, `workloads/`, `analysis/`, `scripts/`, `configs/` with minimal placeholders.
2. **Minimal router proxy:** `POST /v1/chat/completions` forwards to one instance (hard-coded or RR).
3. **Policy interface:** define a `Policy.route(request, states) -> instance_index` contract.
4. **Metrics collector:** poll `/metrics` on an interval; keep `InstanceState` updated.
5. **Add policies:** implement JSQ, P2C, session affinity, prefix-key affinity, scoring.
6. **Workload smoke test:** a tiny local workload generator that sends 10 requests and writes JSONL.
7. **Real workloads:** ShareGPT replay + synthetic RAG; controlled rate and concurrency.
8. **Experiment runner:** config-driven runs; outputs plots-ready data + summary.

## Deliverables (What "Good" Looks Like)

- One command launches the router.
- One command runs a smoke test and produces a results file.
- Every experiment run emits:
  - `run_config.json` (resolved config)
  - `results.jsonl` (one row per request)
  - `summary.json` (aggregates)
  - optional `prometheus_snapshots/` (raw `/metrics` scrapes)

## Implementation Conventions

- Prefer async I/O for router forwarding and metrics scraping (`aiohttp` or `httpx`).
- Timeouts must be explicit (connect, read, total).
- Result rows should include: timestamp, policy, chosen instance, session/user id, prefix key, prompt token estimate (if available), router latency, and any engine-side metrics we can reliably scrape.
- Keep all randomness seeded and recorded in outputs.

## Review Checklist Before Handing Off

- The change is incremental: the system still runs after your patch.
- There is a smoke test command.
- Any breaking change is documented in `docs/AI_SHARED.md`.
- Output formats are stable and versioned if necessary.

## Preferred Handoff Output Format

- Short changelog + file list
- Exact run commands
- Expected outputs (filenames or key lines)

