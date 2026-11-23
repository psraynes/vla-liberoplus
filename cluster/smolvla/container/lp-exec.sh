#!/bin/bash
# Run a command inside the LIBERO-Plus container with GPU rendering, the assets squashfs mounted, and your code on
# PYTHONPATH. Use only on compute nodes (inside sbatch / srun jobs).
#   lp-exec.sh python -c "import libero"            lp-exec.sh python my_eval.py ...
# Settings (environment variables):
#   LP_DIR         image + assets directory           (default /scratch/$USER/smolvla_turing)
#   LP_CODE        checkouts directory                 (default $HOME/smolvla_turing/code; LIBERO-plus must be in it)
#   LP_PYTHONPATH  extra paths for your VLA code       (e.g. "$HOME/smolvla_turing/code/MyVLA")
#   LP_EGL_VENDOR  EGL vendor file                     (default NVIDIA; .../50_mesa.json = CPU fallback)
set -euo pipefail
LP_DIR=${LP_DIR:-/scratch/$USER/smolvla_turing}
LP_CODE=${LP_CODE:-$HOME/smolvla_turing/code}

if ! command -v apptainer > /dev/null; then module load apptainer > /dev/null 2>&1; fi
for f in "$LP_DIR/liberoplus.sif" "$LP_DIR/assets.sqfs"; do
  [ -f "$f" ] || { echo "lp-exec.sh: missing $f" >&2; exit 1; }
done

# Assets are mounted at LIBERO's package-relative location: some LIBERO code (e.g. TableArena via
# xml_path_completion) ignores the config's `assets:` path and always reads <package>/assets.
ASSETS_MNT="$LP_CODE/LIBERO-plus/libero/libero/assets"
[ -d "$ASSETS_MNT" ] || { echo "lp-exec.sh: missing mount point $ASSETS_MNT (run setup_libero_plus.sh)" >&2; exit 1; }

# Render on the NVIDIA GPU only. Without this, glvnd also offers Mesa's software device and MuJoCo picks it first
# (0% GPU, ~4x slower).
export APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES="${LP_EGL_VENDOR:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export APPTAINERENV_PYTHONPATH="$LP_CODE/LIBERO-plus${LP_PYTHONPATH:+:$LP_PYTHONPATH}"
export APPTAINERENV_LIBERO_CONFIG_PATH="$HOME/smolvla_turing/libero_config"
exec apptainer exec --nv \
  -B /scratch \
  -B "$LP_DIR/assets.sqfs:$ASSETS_MNT:image-src=/" \
  "$LP_DIR/liberoplus.sif" "$@"
