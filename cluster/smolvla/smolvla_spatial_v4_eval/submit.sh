#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ACCOUNT=${1:-cs595}
test ! -e submission_ids.txt || { cat submission_ids.txt; exit 2; }
sha256sum -c kit.sha256
bash -n verify_v4.sbatch run_v4.sbatch standard_v4.sbatch
V=$(sbatch --parsable --account="$ACCOUNT" verify_v4.sbatch)
echo "Verify=$V" | tee submission_ids.txt
E=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$V" --array=0-1%2 run_v4.sbatch)
echo "CleanArray=$E" | tee -a submission_ids.txt
S=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:$V" standard_v4.sbatch)
echo "Standard=$S" | tee -a submission_ids.txt
squeue --me
