# 2-Week Timeline (Presentation Due Feb 26, 2026)

This is the execution plan leading up to the presentation. Keep it updated as tasks complete.

## Week 1 (Feb 12–Feb 18)

- M0: Create repo scaffold for router/workloads/analysis.
- M1: Minimal router proxy (Round-Robin) working end-to-end.
- M2: Metrics scraper working against both vLLM instances.
- Smoke workload produces `results.jsonl` + `summary.json`.

Deliverable by end of Week 1:
- A short live demo: `curl` -> router -> vLLM response.
- One slide updated with real metrics screenshots or a small TTFT sample.

## Week 2 (Feb 19–Feb 25)

- Implement remaining policies (JSQ, P2C, Session affinity, Prefix-key affinity, Scoring).
- Implement synthetic RAG workload and (optionally) ShareGPT replay.
- Run first comparison experiment and generate at least:
  - TTFT CDF plot
  - cache hit rate bar chart

Deliverable by end of Week 2:
- Deck updated with 1–2 real plots and a clear conclusion slide (even if preliminary).

## Final Day (Feb 26)

- Re-run the most important experiment once to confirm results.
- Freeze plots and update the conclusion/limitations slides.

