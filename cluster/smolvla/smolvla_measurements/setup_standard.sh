#!/usr/bin/env bash
# Prepare the original, 40-task LIBERO benchmark alongside the existing LIBERO+ checkout.
set -euo pipefail

ROOT="$HOME/smolvla_turing"
REPO="$ROOT/code/LIBERO-standard"
COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01

mkdir -p "$ROOT/code" "$ROOT/libero_standard_config"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$REPO"
fi
git -C "$REPO" fetch --quiet origin "$COMMIT"
git -C "$REPO" checkout --quiet --detach "$COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"

PKG="$REPO/libero/libero"
for path in "$PKG/bddl_files" "$PKG/init_files"; do
  test -d "$path" || { echo "Missing original LIBERO path: $path" >&2; exit 1; }
done
mkdir -p "$PKG/assets" "$REPO/libero/datasets"
cat > "$ROOT/libero_standard_config/config.yaml" <<EOF
assets: $PKG/assets
bddl_files: $PKG/bddl_files
benchmark_root: $PKG
datasets: $REPO/libero/datasets
init_states: $PKG/init_files
EOF
printf 'Original LIBERO checkout: %s\nConfig: %s\n' "$COMMIT" "$ROOT/libero_standard_config/config.yaml"
echo 'The existing LIBERO+ checkout and results were not changed.'
