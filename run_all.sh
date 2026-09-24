#!/usr/bin/env bash
# Reproduce every result table and figure in this repository (~30 s on a laptop).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
python src/mc_experiment.py          # Monte Carlo + threshold model -> results/
python src/fig1_causal_loop.py       # -> figures/fig1_causal_loop.png
python src/fig2_coverage.py          # -> figures/fig2_coverage_heatmap.png
python src/fig3_threshold.py         # -> figures/fig3_threshold_fatigue.png
# Quick smoke test with fewer replications:  MC_R=20 ./run_all.sh
