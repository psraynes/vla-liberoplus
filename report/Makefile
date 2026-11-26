# Build the report PDF into build/main.pdf.
#   make            build with Tectonic (install it with: bash install_latex.sh)
#   make latexmk    build with a full TeX Live install instead (pdflatex + bibtex via latexmk)
#   make clean      delete build/

TECTONIC ?= tectonic
SOURCES := main.tex $(wildcard sections/*.tex) references.bib IEEEtran.cls IEEEtran.bst $(wildcard figures/*)

.PHONY: all latexmk clean

all: build/main.pdf

build/main.pdf: $(SOURCES)
	@mkdir -p build
	$(TECTONIC) --keep-logs --reruns 2 --outdir build main.tex

latexmk:
	@mkdir -p build
	latexmk -pdf -interaction=nonstopmode -outdir=build main.tex

clean:
	rm -rf build
