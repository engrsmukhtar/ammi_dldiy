#!/bin/bash

for fname in *.tex; do
  cat internals/before.tex "$fname" internals/after.tex | pdflatex
  pdfcrop texput.pdf
  convert -density 300 texput-crop.pdf -resize 200% texput2.pdf
  pdf2svg texput2.pdf "../${fname/.tex/.svg}"

  rm texput*
done
