# SPARKS Workshop Paper — CVPR 2026

## Files
- `main.tex` — Full paper (8-page CVPR two-column format)
- `references.bib` — BibTeX bibliography
- `figures/` — Place figure files here (see TODOs below)

## Compile
```bash
# Requires LaTeX + CVPR style files
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If you don't have `cvpr.sty`, comment out `\usepackage{cvpr}` in main.tex
and use the official CVPR 2026 workshop template from the organizers.

## TODO Before Submission
1. **Affiliation**: Replace `[Affiliation — TODO before submission]` with actual institution
2. **Email**: Replace `{TODO}@TODO.edu` with actual contact email
3. **Figures to create**:
   - `figures/architecture.pdf` — Pipeline diagram: events → voxel grid → ResNet-50 → dual heads
   - `figures/domain_gap.pdf` — Scatter plot: val vs test errors across checkpoints
   - `figures/leaderboard.png` — Screenshot of final leaderboard (already have this)
4. **SPADES citation URL**: Verify the dataset URL in references.bib
5. **Baseline score**: Confirm the SPADES paper baseline rotation error (stated as 79° / ~1.38 rad in challenge description)
6. **CNN+GRU test scores**: Fill in $E_t$ and $E_r$ columns for the CNN+GRU row in Table 1 if available from submission logs
7. **Author order**: Confirm ordering with all co-authors

## Paper Summary
- **Title**: Event-Driven Spacecraft Pose Estimation via CNN Direct Regression from Voxel Grids
- **Venue**: SPARKS Workshop @ CVPR 2026
- **Length**: 8 pages
- **Key result**: Score 2.063 (Trans 0.4105, Rot 1.6526 rad), rank 12/13
- **Key finding**: Frame-by-frame CNN > temporal CNN+GRU at 10 Hz; timestamp mismatch is main domain gap
