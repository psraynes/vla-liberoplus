# Final report (IEEE conference format)

LaTeX source of the final report, in the two-column IEEE conference format (`IEEEtran` class, `conference` option,
US Letter). The PDF is not tracked; build it from these files.

## Build the PDF
**Linux or WSL (recommended):**

```bash
cd report
bash install_latex.sh     # once: installs the Tectonic LaTeX engine to ~/.local/bin (no sudo)
make                      # writes build/main.pdf
```

- The first `make` downloads the LaTeX packages it needs (a few MB, cached in `~/.cache/Tectonic`), so it needs
  internet access. Later builds work offline.
- `make clean` deletes `build/`.
- If `tectonic` is not found after installing, add `~/.local/bin` to your `PATH` (the install script prints the line
  to add to `~/.bashrc`).
- Until the first `\cite`, the build prints `warning: errors were issued by BibTeX`. That is BibTeX noting there are
  no citations yet; it goes away once something is cited.

**Other options:**
- **Full TeX Live** (e.g. `sudo apt install texlive-latex-extra texlive-fonts-recommended latexmk`, or MacTeX on
  macOS): `make latexmk`.
- **macOS with Homebrew:** `brew install tectonic`, then `make`.
- **Overleaf:** upload the `report/` folder and set `main.tex` as the main document.

## Install script details
`install_latex.sh` downloads Tectonic 0.15.0 (Linux x86-64) from its GitHub release, checks the file against a
SHA-256 recorded in the script, and installs the single `tectonic` binary to `~/.local/bin` (override with
`PREFIX=...`). Tectonic is a self-contained LaTeX engine (XeTeX-based): it fetches LaTeX packages on demand and runs
BibTeX automatically, so no system TeX install is needed. To uninstall, delete `~/.local/bin/tectonic` and
`~/.cache/Tectonic`.

## Layout
| File | Contents |
|---|---|
| `main.tex` | document setup, title, authors; includes the sections in order |
| `sections/*.tex` | one file per section (abstract, keywords, introduction, related work, method, experimental setup, results, discussion, conclusion, acknowledgment), so several people can edit at once |
| `references.bib` | BibTeX entries; only cited entries appear in the reference list |
| `figures/` | figures for the report (`\includegraphics{name}` finds files here) |
| `IEEEtran.cls`, `IEEEtran.bst` | IEEE class v1.8b and bibliography style v1.12, unmodified (see below) |
| `Makefile`, `install_latex.sh` | build and install helpers |

## Notes
- **Do not edit `IEEEtran.cls` or `IEEEtran.bst`.** They are unmodified copies of the IEEE files, distributed under
  the LaTeX Project Public License 1.3, which requires a modified version to be renamed. Change formatting in
  `main.tex` instead.
- `main.tex` loads `\usepackage[T1]{fontenc}` so the text is set in Times, as IEEE requires. Without it the
  Tectonic/XeTeX build silently falls back to a different font.
- Sources of the format files: IEEE conference template (version 2024-06-27) and the `IEEEtranBST2` package.
