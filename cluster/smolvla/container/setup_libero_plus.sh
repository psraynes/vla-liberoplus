#!/bin/bash
# One-time LIBERO-Plus setup on Turing (light: run on the login node).
# Clones LIBERO-Plus at a fixed commit, creates the (empty) assets mount point used by lp-exec.sh, and writes the
# LIBERO-Plus config. Put your own VLA checkout next to it in $LP_CODE and add it via LP_PYTHONPATH.
#   bash setup_libero_plus.sh
set -euo pipefail
LP_CODE=${LP_CODE:-$HOME/smolvla_turing/code}
COMMIT=4976dc30028e805ff8094b55501d532c48fec182   # LIBERO-plus commit used for the TurboVLA results
mkdir -p "$LP_CODE"
[ -d "$LP_CODE/LIBERO-plus/.git" ] || git clone https://github.com/sylvestf/LIBERO-plus.git "$LP_CODE/LIBERO-plus"
git -C "$LP_CODE/LIBERO-plus" checkout --quiet "$COMMIT"
echo "LIBERO-plus at $(git -C "$LP_CODE/LIBERO-plus" rev-parse --short HEAD)"

R=$LP_CODE/LIBERO-plus/libero/libero
mkdir -p "$R/assets" "$HOME/smolvla_turing/libero_config"
cat > "$HOME/smolvla_turing/libero_config/config.yaml" <<EOF
assets: $R/assets
bddl_files: $R/bddl_files
benchmark_root: $R
datasets: $R/../datasets
init_states: $R/init_files
EOF
echo "wrote $HOME/smolvla_turing/libero_config/config.yaml (assets are mounted at $R/assets by lp-exec.sh)"
