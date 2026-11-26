#!/usr/bin/env bash
# Install the LaTeX engine used to build the report: Tectonic (a self-contained TeX engine that downloads the
# LaTeX packages it needs on first use and runs BibTeX itself). No sudo needed.
#
# Usage:   bash report/install_latex.sh            # installs to ~/.local/bin
#          PREFIX=/some/dir bash report/install_latex.sh
#
# Linux x86-64 only (this includes WSL). On other systems see report/README.md for alternatives
# (Homebrew, a full TeX Live install, or Overleaf).
set -euo pipefail

VERSION="0.15.0"
ASSET="tectonic-${VERSION}-x86_64-unknown-linux-musl.tar.gz"
URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${VERSION}/${ASSET}"
# SHA-256 of the release asset, recorded when this script was written. If GitHub ever serves a different file,
# the install stops instead of running it.
SHA256="dfb82876f2986862996e564fa507a9e576e0c1e3bee63c2c1bd677c2543e6407"
PREFIX="${PREFIX:-$HOME/.local/bin}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This script supports Linux x86-64 (including WSL) only; found $(uname -s) $(uname -m)." >&2
    echo "See report/README.md for other ways to build the PDF." >&2
    exit 1
fi

if command -v tectonic >/dev/null 2>&1 && tectonic --version 2>/dev/null | grep -q "Tectonic ${VERSION}"; then
    echo "Tectonic ${VERSION} is already installed at $(command -v tectonic)."
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading ${ASSET} ..."
curl -fsSL --proto '=https' --tlsv1.2 -o "$tmp/$ASSET" "$URL"

echo "Verifying checksum ..."
echo "${SHA256}  $tmp/$ASSET" | sha256sum -c --quiet -

tar -xzf "$tmp/$ASSET" -C "$tmp" tectonic
mkdir -p "$PREFIX"
install -m 0755 "$tmp/tectonic" "$PREFIX/tectonic"
echo "Installed $("$PREFIX/tectonic" --version) to $PREFIX/tectonic"

case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *) echo "Note: $PREFIX is not on your PATH. Add this line to ~/.bashrc, then open a new terminal:"
       echo "    export PATH=\"$PREFIX:\$PATH\"" ;;
esac
