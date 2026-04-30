# Overleaf submission folder

Upload this folder's contents to Overleaf and set `main.tex` as the root document.

Included files:

- `main.tex`: paper populated from `deliverables/paper/main.tex`, updated to use `neurips_2026.sty`
- `checklist.tex`: filled NeurIPS checklist, included by `main.tex`
- `references.bib`: bibliography copied from the main paper
- `figures/`: copied PNG figures used by the paper, plus the extra generated CDF figure
- `tables/`: copied table exports in `.tex`, `.csv`, and `.md` formats
- `neurips_2026.sty`, `neurips_2026.tex`: formatting files you added

`main.tex` currently uses NeurIPS 2026 submission mode, which anonymizes authors and adds line numbers. To make a named local draft, change `\usepackage{neurips_2026}` to `\usepackage[preprint]{neurips_2026}` near the top of `main.tex`.
