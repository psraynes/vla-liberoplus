#!/usr/bin/env bash
# Run against the original LIBERO task registry in the existing SmolVLA container.
set -euo pipefail
source "$HOME/smolvla_turing/eval_kit/config.sh"
if ! command -v apptainer >/dev/null; then module load apptainer >/dev/null 2>&1; fi

STD_REPO="$HOME/smolvla_turing/code/LIBERO-standard"
STD_PKG="$STD_REPO/libero/libero"
for path in "$LP_DIR/liberoplus.sif" "$LP_DIR/assets.sqfs" \
            "$STD_REPO/.git" "$STD_PKG/bddl_files" "$STD_PKG/init_files" \
            "$HOME/smolvla_turing/libero_standard_config/config.yaml"; do
  test -e "$path" || { echo "Missing standard benchmark setup: $path" >&2; exit 1; }
done
export APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export APPTAINERENV_PYTHONPATH="$STD_REPO:$HOME/smolvla_turing/eval_kit"
export APPTAINERENV_LIBERO_CONFIG_PATH="$HOME/smolvla_turing/libero_standard_config"
exec apptainer exec --nv -B /scratch \
  -B "$LP_DIR/assets.sqfs:$STD_PKG/assets:image-src=/" \
  "$LP_DIR/liberoplus.sif" python -u "$@"
