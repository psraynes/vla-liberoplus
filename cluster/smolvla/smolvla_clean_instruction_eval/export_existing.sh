#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$HOME/smolvla_turing"
(cd "$ROOT/smolvla_full_eval" && python3 report_full.py > full_report.txt)
(cd "$ROOT/smolvla_rollout_eval" && python3 report_combined_v3.py)
tar -czf smolvla_existing_per_task_results.tar.gz \
  -C "$ROOT/smolvla_full_eval" smolvla_base_vs_combined_v3_full_10030.csv full_report.txt \
  -C "$ROOT/smolvla_rollout_eval" paired_rollouts_four_models.csv combined_v3_results_table.txt
echo "Saved $PWD/smolvla_existing_per_task_results.tar.gz"
