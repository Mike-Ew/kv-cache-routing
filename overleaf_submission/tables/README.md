# Generated Analysis Outputs

Generated from local result JSON files under `code/results`.

Important interpretation caveat: the current runs use two vLLM instances sharing one GPU, so absolute latency includes GPU contention. Use these outputs for relative policy comparisons.

## Inputs

- Exp2 trials: 18 summary files
- Exp3 grid points: 72 summary files

## Figures

- `../figures/system_architecture.png`
- `../figures/exp2_latency_cdf.png`
- `../figures/exp2_policy_comparison.png`
- `../figures/exp3_lca_tradeoff.png`
- `../figures/exp3_rag_heatmaps.png`
- `../figures/exp3_sharegpt_heatmaps.png`

## Tables

- `exp1_policy_comparison.tex`
- `exp1_per_trial.tex`
- `exp2_policy_summary.csv`
- `exp2_policy_summary.md`
- `exp2_policy_summary.tex`
- `exp2_trial_summary.csv`
- `exp3_best_balanced_lca.csv`
- `exp3_best_balanced_lca.md`
- `exp3_best_balanced_lca.tex`
- `exp3_grid_summary.csv`
