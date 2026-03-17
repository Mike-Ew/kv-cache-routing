# Speaker Notes

Use this as a lightweight script. Keep it short and update weekly as results come in.

## Slide: Problem

- Emphasize that KV caches are per-instance, so naive balancing wastes cache locality.

## Slide: Key Idea

- State the tradeoff: cache affinity improves TTFT but can worsen queueing delay.

## Slide: System Architecture

- Call out that we measure TTFT from vLLM `/metrics` (engine-side), not just proxy timers.

## Slide: Routing Policies

- Explain that load-only baselines isolate classic load balancing behavior.
- Cache-aware policies represent increasing sophistication.

## Slide: Hardware / Setup

- Be explicit: 2 vLLM instances share one GPU; focus on relative comparisons.

## Slide: Workloads

- ShareGPT: multi-turn locality.
- Synthetic RAG: cross-user shared prefix locality.

## Slide: Experiment Plan

- Each experiment corresponds to a research question (RQ1–RQ3).

## Slide: Risks + Mitigations

- Mention staleness explicitly as a unique part of the project.

